import type { AgentServer } from "@/agent/server";
import type {
  EmitFunction,
  PluginConfig,
  PluginContext,
  PluginDefinition,
  PluginDependencies,
  PluginEvent,
  PluginEventsDefinition,
  PluginInterceptorFunction,
  ReadonlyPluginContext,
  WritablePluginContext,
} from "@/plugins/definition";
import { PluginApiBase } from "@/plugins/definition";
import { AsyncQueue } from "@/shared/async-queue";
import { klona } from "@/shared/klona";
import { newId } from "@/shared/prefixed-id";
import { equal } from "@/shared/stable-equal";
import type { SerializableValue } from "@/shared/stable-serialize";

type PluginExternalInterceptor = {
  server: PluginServer<PluginDefinition>;
  interceptor: PluginInterceptorFunction;
};

// - Server
export class PluginServer<const Definition extends PluginDefinition> {
  #agent: AgentServer;
  #definition: Definition;
  #config: PluginConfig<Definition["config"], "output">;
  #context: PluginContext<Definition["context"], "output">;
  #contextListeners = new Set<{
    selector: (context: PluginContext<Definition["context"], "output">) => SerializableValue;
    callback: (newValue: SerializableValue, oldValue: SerializableValue) => void;
    lastValue: SerializableValue;
  }>();
  #externalInterceptors: PluginExternalInterceptor[] = [];
  #queue: AsyncQueue<PluginEvent<PluginEventsDefinition, "output">> = new AsyncQueue<
    PluginEvent<PluginEventsDefinition, "output">
  >();
  #servicesQueues: AsyncQueue<PluginEvent<PluginEventsDefinition, "output">>[] = [];
  api: PluginApiBase<Definition>;

  constructor(
    agent: AgentServer,
    def: Definition,
    config: PluginConfig<Definition["config"], "output">,
  ) {
    this.#agent = agent;
    this.#definition = def;
    this.#config = config;
    this.#context = def.context.schema.parse(def.context.initial);

    // Initialize api if defined
    const ApiClass =
      this.#definition.api && "implementation" in this.#definition.api
        ? this.#definition.api.implementation(PluginApiBase, this.#definition.api.schema)
        : PluginApiBase;
    this.api = new ApiClass({
      config: this.#config,
      context: this.#createReadonlyContext(),
      emit: this.emit.bind(this) as EmitFunction,
    }) as PluginApiBase<Definition>;
  }

  // Create read-only context with onChange and get
  #createReadonlyContext(): ReadonlyPluginContext<PluginContext<Definition["context"], "output">> {
    return {
      onChange: this.#onContextChange.bind(this),
      get: this.#getContext.bind(this),
    };
  }

  // Create writable context for effects
  #createWritableContext(): WritablePluginContext<PluginContext<Definition["context"], "output">> {
    return {
      onChange: this.#onContextChange.bind(this),
      get: this.#getContext.bind(this),
      set: this.#setContext.bind(this),
    };
  }

  // Context getter - returns a cloned snapshot
  #getContext(): PluginContext<Definition["context"], "output"> {
    return klona(this.#context);
  }

  // Context setter
  #setContext<K extends keyof PluginContext<Definition["context"], "output">>(
    key: K,
    valueOrUpdater:
      | PluginContext<Definition["context"], "output">[K]
      | ((
          prev: PluginContext<Definition["context"], "output">[K],
        ) => PluginContext<Definition["context"], "output">[K]),
  ): void {
    const oldValue = klona(this.#context);
    const newKeyValue =
      typeof valueOrUpdater === "function"
        ? (
            valueOrUpdater as (
              prev: PluginContext<Definition["context"], "output">[K],
            ) => PluginContext<Definition["context"], "output">[K]
          )(this.#context[key])
        : valueOrUpdater;

    this.#context[key] = klona(newKeyValue);

    // Notify listeners
    this.#notifyContextListeners(oldValue);
  }

  // Subscribe to context changes
  #onContextChange<R extends SerializableValue>(
    selector: (context: PluginContext<Definition["context"], "output">) => R,
    callback: (newValue: R, oldValue: R) => void,
  ): () => void {
    const listener = {
      selector,
      callback: callback as (newValue: SerializableValue, oldValue: SerializableValue) => void,
      lastValue: selector(this.#context),
    };

    this.#contextListeners.add(listener);

    // Return unsubscribe function
    return () => {
      this.#contextListeners.delete(listener);
    };
  }

  // Notify all listeners
  #notifyContextListeners(oldContext: PluginContext<Definition["context"], "output">): void {
    for (const listener of this.#contextListeners) {
      const newSelectedValue = listener.selector(this.#context);
      const oldSelectedValue = listener.selector(oldContext);
      // if (Array.isArray(newSelectedValue))
      //   console.log(
      //     "🐳",
      //     equal(newSelectedValue, oldSelectedValue),
      //     newSelectedValue,
      //     oldSelectedValue,
      //   );

      // Only call if value actually changed
      if (!equal(newSelectedValue, oldSelectedValue)) {
        listener.callback(newSelectedValue, oldSelectedValue);
        listener.lastValue = newSelectedValue;
      }
    }
  }

  // Helper method to call onError lifecycle hook
  async #callOnErrorHook(error: unknown): Promise<void> {
    if (this.#definition.lifecycle?.onError) {
      try {
        await this.#definition.lifecycle.onError({
          config: this.#config,
          context: this.#createWritableContext(),
          emit: this.emit.bind(this) as EmitFunction,
          error,
        });
      } catch (errorHandlerError) {
        console.error(
          `Error in onError lifecycle hook for plugin ${this.#definition.name}:`,
          errorHandlerError,
        );
      }
    }
  }

  #buildDependencies() {
    const dependencies: Record<string, unknown> = {};

    for (const [depName, depDef] of Object.entries(this.#definition.dependencies ?? {})) {
      const depServer = this.#agent.plugins[depName];
      if (!depServer) continue;

      dependencies[depName] = {
        events: depDef.events,
        api: depServer.api,
        config: depServer.#config,
        context: depServer.#createReadonlyContext(),
        emit: depServer.emit.bind(depServer),
      };
    }

    return dependencies as PluginDependencies<Definition["dependencies"]>;
  }

  init() {
    // 1. Initialize services
    const dependencies = this.#buildDependencies();

    for (const service of Object.values(this.#definition.services ?? {}) ?? []) {
      const queue = new AsyncQueue<PluginEvent<PluginEventsDefinition, "output">>();
      this.#servicesQueues.push(queue);

      // Create a single readonly context that always returns fresh values
      const readonlyContext = this.#createReadonlyContext();

      service({
        agent: this.#agent,
        queue: queue[Symbol.asyncIterator](),
        config: this.#config,
        context: readonlyContext,
        api: this.api,
        emit: this.emit.bind(this) as EmitFunction,
        dependencies,
      });
    }

    // 2. Register interceptors with dependencies
    for (const interceptor of Object.values(this.#definition.interceptors ?? {})) {
      // Register this interceptor with each dependency it intercepts
      for (const depName of Object.keys(this.#definition.dependencies ?? {})) {
        const dependentServer = this.#agent.plugins[depName];
        if (dependentServer) {
          dependentServer.registerExternalInterceptor({
            server: this as unknown as PluginServer<PluginDefinition>,
            interceptor,
          });
        }
      }
    }
  }

  emit(event: PluginEvent<Definition["events"], "input">) {
    // Ensure the event type exists
    const eventDefinition = this.#definition?.events?.[event.type];
    if (!eventDefinition) throw new Error(`Event of type '${event.type}' not found.`);

    // Validate the event data
    if ("data" in event && eventDefinition.dataSchema) {
      const validation = eventDefinition.dataSchema?.safeParse(event.data);
      if (!validation.success)
        throw new Error(`Event '${event.type}' data is invalid: ${validation.error.message}.`);
    } else if ("data" in event) {
      throw new Error(`Event '${event.type}' provided unexpected data.`);
    }

    // Generate an id for the event
    const id = newId("event");
    const outputEvent = { id, ...event };

    // Append to queue
    if (event.urgent) this.#queue.pushFirst(outputEvent);
    else this.#queue.push(outputEvent);

    // Return the id
    return id;
  }

  async start() {
    // Run start lifecycle hook
    if (this.#definition.lifecycle?.onStart) {
      try {
        await this.#definition.lifecycle.onStart({
          config: this.#config,
          context: this.#createWritableContext(),
          emit: this.emit.bind(this) as EmitFunction,
        });
      } catch (error) {
        console.error(
          `Error in onStart lifecycle hook for plugin ${this.#definition.name}:`,
          error,
        );
        // Call onError hook if it exists
        await this.#callOnErrorHook(error);
        throw error;
      }
    }

    // Run the queue
    for await (let event of this.#queue) {
      try {
        // if (
        //   event.type !== "user.audio-chunk" &&
        //   event.type !== "user.voice-chunk" &&
        //   event.type !== "agent.voice-chunk"
        // )
        //   console.log("🐳", event);

        // 1. Run external interceptors
        let isDropped = false;
        for (const { interceptor, server } of this.#externalInterceptors) {
          const drop = (_reason: string) => (isDropped = true);
          const next = (newEvent: PluginEvent<PluginEventsDefinition, "output">) => {
            event = newEvent;
          };

          // biome-ignore lint/nursery/noAwaitInLoop: sequential execution expected here
          await interceptor({
            event,
            next,
            drop,
            dependency: {
              name: this.#definition.name,
              events: this.#definition.events,
              // @ts-expect-error
              api: this.api,
              // @ts-expect-error
              config: this.#config,
              // @ts-expect-error
              context: this.#createReadonlyContext(),
              emit: this.emit.bind(this) as EmitFunction,
            },
            current: {
              emit: server.emit.bind(server) as EmitFunction,
              context: server.#createReadonlyContext(),
              api: server.api,
              config: server.#config,
            },
          });
          if (isDropped) break;
        }
        if (isDropped) continue;

        // 2. Run effects
        const dependencies = this.#buildDependencies();
        for (const effect of Object.values(this.#definition.effects ?? {})) {
          // biome-ignore lint/nursery/noAwaitInLoop: sequential execution expected here
          await effect({
            agent: this.#agent,
            event: klona(event),
            config: this.#config,
            context: this.#createWritableContext(),
            api: this.api,
            emit: this.emit.bind(this) as EmitFunction<Definition["events"]>,
            dependencies,
          });
        }

        // 2. Feed services' queues
        for (const queue of this.#servicesQueues) {
          queue.push(klona(event));
        }
      } catch (error) {
        console.error(`Error processing event in plugin ${this.#definition.name}:`, error);
        // Call onError hook if it exists
        await this.#callOnErrorHook(error);
      }
    }
  }

  registerExternalInterceptor(interceptor: PluginExternalInterceptor) {
    this.#externalInterceptors.push(interceptor);
  }

  async stop() {
    // Run stop lifecycle hook
    if (this.#definition.lifecycle?.onStop) {
      try {
        await this.#definition.lifecycle.onStop({
          config: this.#config,
          context: this.#createWritableContext(),
          emit: this.emit.bind(this) as EmitFunction,
        });
      } catch (error) {
        console.error(`Error in onStop lifecycle hook for plugin ${this.#definition.name}:`, error);
        // Call onError hook if it exists
        await this.#callOnErrorHook(error);
        throw error;
      }
    }
  }
}
