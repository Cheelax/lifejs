import { ensureServer } from "@/shared/ensure-server";

ensureServer("transport.server");

import { z } from "zod";
import { TransportCommon } from "./common";
import type { BaseServerTransportProvider } from "./providers/base/server";
import { LiveKitServerTransport, livekitServerConfigSchema } from "./providers/livekit/server";

// Providers
export const serverTransportProviders = {
  livekit: { class: LiveKitServerTransport, configSchema: livekitServerConfigSchema },
} as const;
export type ServerTransportProvider =
  (typeof serverTransportProviders)[keyof typeof serverTransportProviders]["class"];

// Config
export type ServerTransportProviderConfig<T extends "input" | "output"> = {
  [K in keyof typeof serverTransportProviders]: { provider: K } & (T extends "input"
    ? z.input<(typeof serverTransportProviders)[K]["configSchema"]>
    : z.output<(typeof serverTransportProviders)[K]["configSchema"]>);
}[keyof typeof serverTransportProviders];
export const serverTransportProviderConfigSchema = z.discriminatedUnion(
  "provider",
  Object.entries(serverTransportProviders).map(([key, { configSchema }]) =>
    configSchema.extend({ provider: z.literal(key) }),
  ) as unknown as [
    z.ZodObject<{ provider: z.ZodString }>,
    ...z.ZodObject<{ provider: z.ZodString }>[],
  ],
);

// Transport
export class TransportServer extends TransportCommon {
  _provider: BaseServerTransportProvider<z.AnyZodObject>;

  constructor(config: ServerTransportProviderConfig<"output">) {
    super();
    const serverTransportProvider = serverTransportProviders[config.provider];
    this._provider = new serverTransportProvider.class(config);
  }

  // Proxy base methods from the provider for simpler usage
  on: BaseServerTransportProvider<z.AnyZodObject>["on"] = (...args) => this._provider.on(...args);
  joinRoom: BaseServerTransportProvider<z.AnyZodObject>["joinRoom"] = (...args) =>
    this._provider.joinRoom(...args);
  leaveRoom: BaseServerTransportProvider<z.AnyZodObject>["leaveRoom"] = (...args) =>
    this._provider.leaveRoom(...args);
  streamText: BaseServerTransportProvider<z.AnyZodObject>["streamText"] = (...args) =>
    this._provider.streamText(...args);
  receiveStreamText: BaseServerTransportProvider<z.AnyZodObject>["receiveStreamText"] = (...args) =>
    this._provider.receiveStreamText(...args);
  streamAudioChunk: BaseServerTransportProvider<z.AnyZodObject>["streamAudioChunk"] = (...args) =>
    this._provider.streamAudioChunk(...args);
}
