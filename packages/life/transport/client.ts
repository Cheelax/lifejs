import { z } from "zod";
import { TransportCommon } from "./common";
import type { BaseClientTransportProvider } from "./providers/base/client";
import {
  LiveKitClientTransportProvider,
  livekitClientConfigSchema,
} from "./providers/livekit/client";

// Providers
export const clientTransportProviders = {
  livekit: { class: LiveKitClientTransportProvider, configSchema: livekitClientConfigSchema },
} as const;
export type ClientTransportProvider =
  (typeof clientTransportProviders)[keyof typeof clientTransportProviders]["class"];

// Config
export type ClientTransportProviderConfig<T extends "input" | "output"> = {
  [K in keyof typeof clientTransportProviders]: { provider: K } & (T extends "input"
    ? z.input<(typeof clientTransportProviders)[K]["configSchema"]>
    : z.output<(typeof clientTransportProviders)[K]["configSchema"]>);
}[keyof typeof clientTransportProviders];
export const clientTransportProviderConfigSchema = z.discriminatedUnion(
  "provider",
  Object.entries(clientTransportProviders).map(([key, { configSchema }]) =>
    configSchema.extend({ provider: z.literal(key) }),
  ) as unknown as [
    z.ZodObject<{ provider: z.ZodString }>,
    ...z.ZodObject<{ provider: z.ZodString }>[],
  ],
);

// Transport
export class TransportClient extends TransportCommon {
  _provider: BaseClientTransportProvider<z.AnyZodObject>;

  constructor(config: ClientTransportProviderConfig<"output">) {
    super();
    const clientTransportProvider = clientTransportProviders[config.provider];
    this._provider = new clientTransportProvider.class(config);
  }

  // Proxy base methods from the provider for simpler usage
  on: BaseClientTransportProvider<z.AnyZodObject>["on"] = (...args) => this._provider.on(...args);
  joinRoom: BaseClientTransportProvider<z.AnyZodObject>["joinRoom"] = (...args) =>
    this._provider.joinRoom(...args);
  leaveRoom: BaseClientTransportProvider<z.AnyZodObject>["leaveRoom"] = (...args) =>
    this._provider.leaveRoom(...args);
  streamText: BaseClientTransportProvider<z.AnyZodObject>["streamText"] = (...args) =>
    this._provider.streamText(...args);
  receiveStreamText: BaseClientTransportProvider<z.AnyZodObject>["receiveStreamText"] = (...args) =>
    this._provider.receiveStreamText(...args);
  enableMicrophone: BaseClientTransportProvider<z.AnyZodObject>["enableMicrophone"] = (...args) =>
    this._provider.enableMicrophone(...args);
  playAudio: BaseClientTransportProvider<z.AnyZodObject>["playAudio"] = (...args) =>
    this._provider.playAudio(...args);
}
