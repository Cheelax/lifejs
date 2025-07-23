import z from "zod";
import {
  type ClientTransportProviderConfig,
  clientTransportProviderConfigSchema,
} from "@/transport/client";
import type { ServerConfig } from "./server";

export const clientConfigDefinitionSchema = z.object({
  transport: clientTransportProviderConfigSchema,
});

export type ClientConfig = {
  transport: ClientTransportProviderConfig<"output">;
};

export function serverToClientConfig<S extends ServerConfig<"output">>(serverConfig: S) {
  return clientConfigDefinitionSchema.parse(serverConfig) as ClientConfig;
}
