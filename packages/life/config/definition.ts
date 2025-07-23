import { type ServerConfig, serverConfigDefinitionSchema } from "./server";

// Config is defined server-side (hence why we use the server config schema) and
// then transformed into a client-side config object by the compiler.
export function defineConfig(def: ServerConfig<"input">) {
  const parsedConfig = serverConfigDefinitionSchema.parse(def);
  return { raw: def, withDefaults: parsedConfig };
}
