import { z } from "zod";
import { defineConfig, type ServerConfig } from "@/config/server";
import {
  definePlugin,
  type PluginConfig,
  type PluginDefinition,
  type PluginDependenciesDefinition,
  type PluginDependencyDefinition,
} from "@/plugins/definition";

export type AgentDefinition = {
  name: string;
  config: ServerConfig<"output">;
  plugins: Record<string, PluginDefinition>;
  pluginConfigs: Record<string, unknown>;
};

type WithPluginsMethods<
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  Builder extends AgentDefinitionBuilder<any, any>,
  PluginsDefs extends PluginDependenciesDefinition,
  ExcludedMethods extends string,
> = Builder & {
  [K in keyof PluginsDefs]: K extends string
    ? (
        config: PluginConfig<Extract<PluginsDefs[K], { name: K }>["config"], "input">,
      ) => Omit<WithPluginsMethods<Builder, PluginsDefs, ExcludedMethods | K>, ExcludedMethods | K>
    : never;
};

export class AgentDefinitionBuilder<
  const Definition extends AgentDefinition,
  ExcludedMethods extends string = never,
> {
  _definition: Definition;

  constructor(definition: Definition) {
    this._definition = definition;
  }

  config(params: ServerConfig<"input">) {
    // Create a new builder instance with the provided config
    const builder = new AgentDefinitionBuilder({
      ...this._definition,
      config: defineConfig(params).withDefaults,
    }) as AgentDefinitionBuilder<Definition, ExcludedMethods | "config">;

    // Return the new builder with the plugins methods, minus excluded methods
    return this.#withPluginsMethods(builder, this._definition.plugins) as Omit<
      WithPluginsMethods<typeof builder, Definition["plugins"], ExcludedMethods | "config">,
      ExcludedMethods | "config"
    >;
  }

  plugins<const Plugins extends readonly { _definition: PluginDefinition }[]>(plugins: Plugins) {
    // Convert array of plugin builders to dependencies definition
    const defs: PluginDependenciesDefinition = {};
    for (const plugin of plugins) defs[plugin._definition.name] = plugin._definition;

    // Type to extract dependency definition from array of plugins
    type ExtractedDefs = {
      [K in Plugins[number] as K["_definition"]["name"]]: {
        name: K["_definition"]["name"];
        events: K["_definition"]["events"];
        config: K["_definition"]["config"];
        context: K["_definition"]["context"];
        api: K["_definition"]["api"];
      };
    };

    // Create a new builder instance with the provided plugins
    const builder = new AgentDefinitionBuilder({
      ...this._definition,
      plugins: defs,
    }) as unknown as AgentDefinitionBuilder<
      Definition & { plugins: ExtractedDefs },
      ExcludedMethods | "plugins"
    >;

    // Return the new builder with the plugins methods, minus excluded methods
    return this.#withPluginsMethods(builder, defs) as Omit<
      WithPluginsMethods<typeof builder, ExtractedDefs, ExcludedMethods | "plugins">,
      ExcludedMethods | "plugins"
    >;
  }

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  #withPluginsMethods<Builder extends AgentDefinitionBuilder<any, any>>(
    builder: Builder,
    plugins: PluginDependenciesDefinition,
  ) {
    for (const plugin of Object.values(plugins)) {
      Object.assign(builder, {
        [plugin.name]: this.#pluginMethod(plugin, plugins),
      });
    }
    return builder;
  }

  #pluginMethod(plugin: PluginDependencyDefinition, plugins: PluginDependenciesDefinition) {
    return (config: z.input<PluginDefinition["config"]>): unknown => {
      const builder = new AgentDefinitionBuilder({
        ...this._definition,
        pluginConfigs: {
          ...((this._definition as Definition).pluginConfigs ?? {}),
          [plugin.name]: plugin.config.parse(config),
        },
      });
      return this.#withPluginsMethods(builder, plugins);
    };
  }
}

export function defineAgent<const Name extends string>(name: Name) {
  return new AgentDefinitionBuilder({
    name,
    config: defineConfig({}).withDefaults as ServerConfig<"output">,
    plugins: {},
    pluginConfigs: {},
  });
}

const collectionsPlugin = definePlugin("collections").config(
  z.object({
    collections: z.array(
      z.object({
        name: z.string(),
        description: z.string(),
      }),
    ),
  }),
);

const agent = defineAgent("example").plugins([collectionsPlugin]).config({});

console.log(agent);
