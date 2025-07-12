import { z } from "zod";
import { AnthropicLLM, anthropicLLMConfigSchema } from "./providers/anthropic";
import { MistralLLM, mistralLLMConfigSchema } from "./providers/mistral";
import { OpenAILLM, openAILLMConfigSchema } from "./providers/openai";
import { XaiLLM, xaiLLMConfigSchema } from "./providers/xai";

// Providers
export const llmProviders = {
  anthropic: { class: AnthropicLLM, configSchema: anthropicLLMConfigSchema },
  mistral: { class: MistralLLM, configSchema: mistralLLMConfigSchema },
  openai: { class: OpenAILLM, configSchema: openAILLMConfigSchema },
  xai: { class: XaiLLM, configSchema: xaiLLMConfigSchema },
} as const;

export type LLMProvider = (typeof llmProviders)[keyof typeof llmProviders]["class"];

// Config
export type LLMProviderConfig<T extends "input" | "output"> = {
  [K in keyof typeof llmProviders]: { provider: K } & (T extends "input"
    ? z.input<(typeof llmProviders)[K]["configSchema"]>
    : z.output<(typeof llmProviders)[K]["configSchema"]>);
}[keyof typeof llmProviders];

export const llmProviderConfigSchema = z.discriminatedUnion(
  "provider",
  Object.entries(llmProviders).map(([key, { configSchema }]) =>
    configSchema.extend({ provider: z.literal(key) }),
  ) as unknown as [
    z.ZodObject<{ provider: z.ZodString }>,
    ...z.ZodObject<{ provider: z.ZodString }>[],
  ],
);
