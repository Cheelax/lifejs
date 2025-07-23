import { z } from "zod";
import type { AgentServer } from "@/agent/server";
import type { AsyncQueue } from "@/shared/async-queue";
import type { SerializableValue } from "@/shared/stable-serialize";

/** 
@dev Because this plugin builder uses a lot of union and nested types, it might be a little slow with 
Intellisense. If in the future that became an issue, a simple solution could be to lose some typesafety 
depth by replacing many nested types with 'any' (not the top-level ones though, so preserving the same 
typesafe experience).
*/

// - Common
export type EmitFunction<EventsDef extends PluginEventsDefinition = PluginEventsDefinition> = (
  event: PluginEvent<EventsDef, "input">,
) => string;

// - Dependencies
export type PluginDependencyDefinition = Pick<
  PluginDefinition,
  "name" | "events" | "config" | "context" | "api"
>;
export type PluginDependenciesDefinition = Record<string, PluginDependencyDefinition>;
export type PluginDependencies<Defs extends PluginDependenciesDefinition> = {
  [K in keyof Defs]: {
    name: Defs[K]["name"];
    events: Defs[K]["events"];
    config: Defs[K]["config"];
    context: Defs[K]["context"];
    api: Defs[K]["api"];
    emit: EmitFunction<Defs[K]["events"]>;
  };
};

// - Config
export type PluginConfigDefinition = z.AnyZodObject;
export type PluginConfig<
  Def extends PluginConfigDefinition,
  T extends "input" | "output",
> = Readonly<T extends "input" ? z.input<Def> : z.output<Def>>;

// - Context
export type PluginContextDefinition<Schema extends z.AnyZodObject = z.AnyZodObject> = {
  schema: Schema;
  initial: z.input<Schema>;
};
export type PluginContext<
  Def extends PluginContextDefinition,
  T extends "input" | "output",
> = T extends "input" ? z.input<Def["schema"]> : z.output<Def["schema"]>;

export type ReadonlyPluginContext<T> = {
  onChange<R extends SerializableValue>(
    selector: (context: T) => R,
    callback: (newValue: R, oldValue: R) => void,
  ): () => void; // Returns unsubscribe function
  get(): T; // Returns a cloned snapshot of the context
};

export type WritablePluginContext<T> = {
  onChange<R extends SerializableValue>(
    selector: (context: T) => R,
    callback: (newValue: R, oldValue: R) => void,
  ): () => void; // Returns unsubscribe function
  get(): T; // Returns a cloned snapshot of the context
  set<K extends keyof T>(key: K, valueOrUpdater: T[K] | ((prev: T[K]) => T[K])): void;
};

// - Events
export type PluginEventsDefinition = Record<string, { dataSchema?: z.Schema }>;
export type PluginEvent<EventsDef extends PluginEventsDefinition, T extends "input" | "output"> = {
  [K in keyof EventsDef]: {
    type: K extends string ? K : never;
    urgent?: boolean;
  } & (EventsDef[K]["dataSchema"] extends z.Schema
    ? {
        data: T extends "input"
          ? z.input<EventsDef[K]["dataSchema"]>
          : z.output<EventsDef[K]["dataSchema"]>;
      }
    : // biome-ignore lint/complexity/noBannedTypes: empty object type needed for conditional
      {}) &
    (T extends "output"
      ? { id: string }
      : // biome-ignore lint/complexity/noBannedTypes: empty object type needed for conditional
        {});
}[keyof EventsDef];

// - API
export interface PluginApiDefinition<
  Schema extends z.AnyZodObject = z.AnyZodObject,
  Definition extends PluginDefinition = PluginDefinition,
> {
  schema: Schema;
  implementation: (
    Base: typeof PluginApiBase<Definition>,
    schema: Schema,
  ) => new (
    raw: PluginApiConnector<Definition>,
  ) => PluginApiBase<Definition> & z.TypeOf<Schema>;
}

export type PluginApi<Def extends PluginApiDefinition> = z.TypeOf<Def["schema"]>;

export type PluginApiConnector<Definition extends PluginDefinition> = {
  context: ReadonlyPluginContext<PluginContext<Definition["context"], "output">>;
  config: PluginConfig<Definition["config"], "output">;
  emit: EmitFunction<Definition["events"]>;
};

export class PluginApiBase<Definition extends PluginDefinition = PluginDefinition> {
  raw: PluginApiConnector<Definition>;
  constructor(raw: PluginApiConnector<Definition>) {
    this.raw = raw;
  }
}

// - Lifecycle
export type PluginLifecycle<Definition extends PluginDefinition = PluginDefinition> = {
  onStart?: (params: {
    config: PluginConfig<Definition["config"], "output">;
    context: WritablePluginContext<PluginContext<Definition["context"], "output">>;
    emit: EmitFunction<Definition["events"]>;
  }) => void | Promise<void>;
  onStop?: (params: {
    config: PluginConfig<Definition["config"], "output">;
    context: WritablePluginContext<PluginContext<Definition["context"], "output">>;
    emit: EmitFunction<Definition["events"]>;
  }) => void | Promise<void>;
  onError?: (params: {
    config: PluginConfig<Definition["config"], "output">;
    context: WritablePluginContext<PluginContext<Definition["context"], "output">>;
    emit: EmitFunction<Definition["events"]>;
    error: unknown;
  }) => void | Promise<void>;
};

// - Effects
export type PluginEffectFunction<Definition extends PluginDefinition = PluginDefinition> =
  (params: {
    event: PluginEvent<Definition["events"], "output">;
    agent: AgentServer;
    config: PluginConfig<Definition["config"], "output">;
    context: WritablePluginContext<PluginContext<Definition["context"], "output">>;
    api: PluginApi<Definition["api"]>;
    dependencies: PluginDependencies<Definition["dependencies"]>;
    emit: EmitFunction<Definition["events"]>;
  }) => void | Promise<void>;
export type PluginEffectsDefinition<Definition extends PluginDefinition = PluginDefinition> =
  Record<string, PluginEffectFunction<Definition>>;

// - Services
export type PluginServiceFunction<Definition extends PluginDefinition = PluginDefinition> =
  (params: {
    queue: AsyncQueue<PluginEvent<Definition["events"], "output">>;
    agent: AgentServer;
    config: PluginConfig<Definition["config"], "output">;
    context: ReadonlyPluginContext<PluginContext<Definition["context"], "output">>;
    api: PluginApi<Definition["api"]>;
    dependencies: PluginDependencies<Definition["dependencies"]>;
    emit: EmitFunction<Definition["events"]>;
  }) => void | Promise<void>;
export type PluginServicesDefinition<Definition extends PluginDefinition = PluginDefinition> =
  Record<string, PluginServiceFunction<Definition>>;

// - Interceptors
export type PluginInterceptorFunction<Definition extends PluginDefinition = PluginDefinition> =
  (params: {
    event: PluginEvent<
      PluginDependencies<Definition["dependencies"]>[keyof PluginDependencies<
        Definition["dependencies"]
      >]["events"],
      "output"
    >;
    next: (
      event: PluginEvent<
        PluginDependencies<Definition["dependencies"]>[keyof PluginDependencies<
          Definition["dependencies"]
        >]["events"],
        "output"
      >,
    ) => void;
    drop: (reason: string) => void;
    dependency: PluginDependencies<Definition["dependencies"]>[keyof Definition["dependencies"]] & {
      name: keyof Definition["dependencies"];
    };
    current: {
      emit: EmitFunction<Definition["events"]>;
      context: ReadonlyPluginContext<PluginContext<Definition["context"], "output">>;
      api: PluginApi<Definition["api"]>;
      config: PluginConfig<Definition["config"], "output">;
    };
  }) => void | Promise<void>;
export type PluginInterceptorsDefinition<Definition extends PluginDefinition = PluginDefinition> =
  Record<string, PluginInterceptorFunction<Definition>>;

// - Definition
export interface PluginDefinition {
  name: string;
  dependencies: PluginDependenciesDefinition;
  config: PluginConfigDefinition;
  context: PluginContextDefinition;
  events: PluginEventsDefinition;
  api: PluginApiDefinition;
  lifecycle: PluginLifecycle;
  effects: PluginEffectsDefinition;
  services: PluginServicesDefinition;
  interceptors: PluginInterceptorsDefinition;
}

// - Plugin
export class PluginDefinitionBuilder<
  const Definition extends PluginDefinition,
  EffectKeys extends string = never,
  ServiceKeys extends string = never,
  InterceptorKeys extends string = never,
  ExcludedMethods extends string = never,
> {
  _definition: Definition;

  constructor(def: Definition) {
    this._definition = def;
  }

  dependencies<const Plugins extends { _definition: PluginDefinition }[]>(plugins: Plugins) {
    // Convert array of plugin builders to dependencies definition
    const dependencies: PluginDependenciesDefinition = {};
    for (const plugin of plugins) dependencies[plugin._definition.name] = plugin._definition;

    // Type to extract dependency definition from array of plugins
    type ExtractedDependencies = {
      [K in Plugins[number] as K["_definition"]["name"]]: {
        name: K["_definition"]["name"];
        events: K["_definition"]["events"];
        config: K["_definition"]["config"];
        context: K["_definition"]["context"];
        api: K["_definition"]["api"];
      };
    };

    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      dependencies,
    }) as unknown as PluginDefinitionBuilder<
      Definition & { dependencies: ExtractedDependencies },
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods | "dependencies"
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods | "dependencies">;
  }

  config<const ConfigDef extends PluginConfigDefinition>(config: ConfigDef) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      config,
    }) as PluginDefinitionBuilder<
      Definition & { config: ConfigDef },
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods | "config"
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods | "config">;
  }

  context<Schema extends z.AnyZodObject>(context: PluginContextDefinition<Schema>) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      context,
    }) as PluginDefinitionBuilder<
      Definition & { context: PluginContextDefinition<Schema> },
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods | "context"
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods | "context">;
  }

  events<const EventsDef extends PluginEventsDefinition>(events: EventsDef) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      events,
    }) as PluginDefinitionBuilder<
      Definition & { events: EventsDef },
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods | "events"
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods | "events">;
  }

  api<const Schema extends z.AnyZodObject>(api: PluginApiDefinition<Schema, Definition>) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      api,
    }) as unknown as PluginDefinitionBuilder<
      Definition & { api: PluginApiDefinition<Schema> },
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods | "api"
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods | "api">;
  }

  lifecycle<const LifecycleConfig extends PluginLifecycle<Definition>>(lifecycle: LifecycleConfig) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      lifecycle,
    }) as unknown as PluginDefinitionBuilder<
      Definition,
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods | "lifecycle"
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods | "lifecycle">;
  }

  addEffect<const Name extends string>(name: Name, effect: PluginEffectFunction<Definition>) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      effects: { ...(this._definition.effects ?? {}), [name]: effect },
    }) as PluginDefinitionBuilder<
      Definition,
      EffectKeys | Name,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods>;
  }

  removeEffect<const Name extends EffectKeys>(name: Name) {
    const { [name]: _removed, ...remainingEffects } = this._definition.effects ?? {};
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      effects: remainingEffects,
    }) as unknown as PluginDefinitionBuilder<
      Definition,
      Exclude<EffectKeys, Name>,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods>;
  }

  addService<const Name extends string>(name: Name, service: PluginServiceFunction<Definition>) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      services: { ...(this._definition.services ?? {}), [name]: service },
    }) as PluginDefinitionBuilder<
      Definition,
      EffectKeys,
      ServiceKeys | Name,
      InterceptorKeys,
      ExcludedMethods
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods>;
  }

  removeService<const Name extends ServiceKeys>(name: Name) {
    const { [name]: _removed, ...remainingServices } = this._definition.services ?? {};
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      services: remainingServices,
    }) as unknown as PluginDefinitionBuilder<
      Definition,
      EffectKeys,
      Exclude<ServiceKeys, Name>,
      InterceptorKeys,
      ExcludedMethods
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods>;
  }

  addInterceptor<const Name extends string>(
    name: Name,
    interceptor: PluginInterceptorFunction<Definition>,
  ) {
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      interceptors: { ...(this._definition.interceptors ?? {}), [name]: interceptor },
    }) as PluginDefinitionBuilder<
      Definition,
      EffectKeys,
      ServiceKeys,
      InterceptorKeys | Name,
      ExcludedMethods
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods>;
  }

  removeInterceptor<const Name extends InterceptorKeys>(name: Name) {
    const { [name]: _removed, ...remainingInterceptors } = this._definition.interceptors ?? {};
    const plugin = new PluginDefinitionBuilder({
      ...this._definition,
      interceptors: remainingInterceptors,
    }) as unknown as PluginDefinitionBuilder<
      Definition,
      EffectKeys,
      ServiceKeys,
      Exclude<InterceptorKeys, Name>,
      ExcludedMethods
    >;
    return plugin as Omit<typeof plugin, ExcludedMethods>;
  }

  pick<
    const Options extends {
      events?: Array<keyof Definition["events"]>;
      api?: Array<keyof Definition["api"]["schema"]["shape"]>;
      context?: Array<keyof Definition["context"]["schema"]["shape"]>;
      config?: boolean;
    },
  >(_options: Options) {
    // Pick is now type-only - runtime always returns the full plugin
    // TypeScript will enforce the constraints at compile time
    const pickedDefinition: PluginDefinition = this._definition;

    // Type for the picked definition
    type PickedDefinition = {
      name: Definition["name"];
      config: Options["config"] extends true ? Definition["config"] : never;
      events: Options["events"] extends readonly string[]
        ? Pick<Definition["events"], Options["events"][number]>
        : never;
      context: Options["context"] extends readonly string[]
        ? Omit<Definition["context"], "schema"> & {
            schema: z.ZodObject<
              Pick<Definition["context"]["schema"]["shape"], Options["context"][number]>
            >;
          }
        : never;
      api: Options["api"] extends readonly string[]
        ? Omit<Definition["api"], "schema"> & {
            schema: z.ZodObject<Pick<Definition["api"]["schema"]["shape"], Options["api"][number]>>;
          }
        : never;
      dependencies: never;
      lifecycle: never;
      effects: never;
      services: never;
      interceptors: never;
    };

    return new PluginDefinitionBuilder(pickedDefinition) as unknown as PluginDefinitionBuilder<
      PickedDefinition,
      EffectKeys,
      ServiceKeys,
      InterceptorKeys,
      ExcludedMethods
    >;
  }
}

export function definePlugin<const Name extends string>(name: Name) {
  return new PluginDefinitionBuilder({
    name,
    dependencies: {},
    config: z.object({}),
    context: {
      schema: z.object({}),
      initial: {},
    },
    events: {},
    api: {
      schema: z.object({}),
      implementation: (Base) => class extends Base {},
    },
    lifecycle: {},
    effects: {},
    interceptors: {},
    services: {},
  });
}
