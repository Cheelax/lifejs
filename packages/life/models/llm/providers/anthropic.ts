import type { Message, ToolDefinition } from "@/agent/resources";
import Anthropic from "@anthropic-ai/sdk";
import type {
  MessageParam,
  MessageStreamEvent,
  Tool,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { LLMBase, type LLMGenerateMessageJob } from "../base";

// Config
export const anthropicLLMConfigSchema = z.object({
  apiKey: z.string().default(process.env.ANTHROPIC_API_KEY ?? ""),
  model: z
    .enum([
      "claude-3-5-sonnet-20241022",
      "claude-3-5-haiku-20241022",
      "claude-3-opus-20240229",
      "claude-3-sonnet-20240229",
      "claude-3-haiku-20240307",
    ])
    .default("claude-3-5-sonnet-20241022"),
  temperature: z.number().min(0).max(1).default(0.5),
  maxTokens: z.number().min(1).max(8192).default(4096),
});

// Model
export class AnthropicLLM extends LLMBase<typeof anthropicLLMConfigSchema> {
  #client: Anthropic;

  constructor(config: z.input<typeof anthropicLLMConfigSchema>) {
    super(anthropicLLMConfigSchema, config);
    if (!config.apiKey)
      throw new Error(
        "ANTHROPIC_API_KEY environment variable or config.apiKey must be provided to use this model.",
      );
    this.#client = new Anthropic({ apiKey: config.apiKey });
  }

  /**
   * Format conversion
   */

  #toAnthropicMessage(message: Message): MessageParam {
    if (message.role === "user") {
      return { role: "user", content: message.content };
    }

    if (message.role === "agent") {
      const content: Array<Anthropic.TextBlockParam | Anthropic.ToolUseBlockParam> = [];

      if (message.content) {
        content.push({ type: "text", text: message.content });
      }

      if (message.toolsRequests?.length) {
        for (const request of message.toolsRequests) {
          content.push({
            type: "tool_use",
            id: request.id,
            name: request.name,
            input: request.input,
          });
        }
      }

      return { role: "assistant", content };
    }

    if (message.role === "system") {
      // System messages are handled separately in Anthropic
      return { role: "user", content: message.content };
    }

    if (message.role === "tool-response") {
      return {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: message.toolId,
            content: JSON.stringify(message.toolOutput),
          },
        ],
      };
    }

    return null as never;
  }

  #toAnthropicMessages(messages: Message[]): {
    system?: string;
    messages: MessageParam[];
  } {
    // Extract system messages
    const systemMessages = messages.filter((m) => m.role === "system");
    const system = systemMessages.map((m) => m.content).join("\n");

    // Convert non-system messages
    const anthropicMessages = messages
      .filter((m) => m.role !== "system")
      .map(this.#toAnthropicMessage.bind(this));

    return {
      ...(system ? { system } : {}),
      messages: anthropicMessages,
    };
  }

  #toAnthropicTool(tool: ToolDefinition): Tool {
    return {
      name: tool.name,
      description: tool.description,
      input_schema: zodToJsonSchema(tool.inputSchema) as Tool.InputSchema,
    };
  }

  #toAnthropicTools(tools: ToolDefinition[]): Tool[] {
    return tools.map(this.#toAnthropicTool.bind(this));
  }

  /**
   * Generate a message with job management - returns jobId along with stream
   */
  async generateMessage(
    params: Parameters<typeof LLMBase.prototype.generateMessage>[0],
  ): Promise<LLMGenerateMessageJob> {
    // Create a new job
    const job = this.createGenerateMessageJob();

    // Prepare tools and messages in Anthropic format
    const anthropicTools = params.tools.length > 0 ? this.#toAnthropicTools(params.tools) : undefined;
    const { system, messages: anthropicMessages } = this.#toAnthropicMessages(params.messages);

    try {
      // Create stream with Anthropic SDK
      const stream = await this.#client.messages.create(
        {
          model: this.config.model,
          temperature: this.config.temperature,
          max_tokens: this.config.maxTokens,
          messages: anthropicMessages,
          ...(system ? { system } : {}),
          ...(anthropicTools?.length ? { tools: anthropicTools } : {}),
          stream: true,
        },
        { signal: job.raw.abortController.signal },
      );

      // Start streaming in the background (don't await)
      (async () => {
        try {
          let currentToolUse: { id: string; name: string; input: string } | null = null;
          const completedTools: Array<{ id: string; name: string; input: Record<string, any> }> = [];

          for await (const event of stream) {
            // Ignore events if job was cancelled
            if (job.raw.abortController.signal.aborted) break;

            switch (event.type) {
              case "content_block_start":
                if (event.content_block.type === "tool_use") {
                  // Start accumulating a new tool use
                  currentToolUse = {
                    id: event.content_block.id,
                    name: event.content_block.name,
                    input: "",
                  };
                }
                break;

              case "content_block_delta":
                if (event.delta.type === "text_delta") {
                  // Stream text content
                  job.raw.receiveChunk({ type: "content", content: event.delta.text });
                } else if (event.delta.type === "input_json_delta" && currentToolUse) {
                  // Accumulate tool input
                  currentToolUse.input += event.delta.partial_json;
                }
                break;

              case "content_block_stop":
                if (currentToolUse) {
                  // Parse and store the completed tool use
                  completedTools.push({
                    id: currentToolUse.id,
                    name: currentToolUse.name,
                    input: JSON.parse(currentToolUse.input || "{}"),
                  });
                  currentToolUse = null;
                }
                break;

              case "message_stop": {
                // Send all completed tools at once
                if (completedTools.length > 0) {
                  job.raw.receiveChunk({ type: "tools", tools: completedTools });
                }
                // Mark stream as ended
                job.raw.receiveChunk({ type: "end" });
                break;
              }

              default:
                // Ignore other event types
                break;
            }
          }
        } catch (error) {
          // Handle any streaming errors
          job.raw.receiveChunk({
            type: "error",
            error: error instanceof Error ? error.message : "Unknown streaming error",
          });
        }
      })();

      // Return the job immediately
      return job;
    } catch (error) {
      // Handle initial request errors
      job.raw.receiveChunk({
        type: "error",
        error: error instanceof Error ? error.message : "Failed to create message stream",
      });
      return job;
    }
  }

  async generateObject(
    params: Parameters<typeof LLMBase.prototype.generateObject>[0],
  ): ReturnType<typeof LLMBase.prototype.generateObject> {
    try {
      // Prepare messages in Anthropic format
      const { system, messages: anthropicMessages } = this.#toAnthropicMessages(params.messages);

      // Create a tool that will extract the structured data
      const extractionTool: Tool = {
        name: "extract_structured_data",
        description: "Extract structured data according to the provided schema",
        input_schema: zodToJsonSchema(params.schema) as Tool.InputSchema,
      };

      // Generate with tool use
      const response = await this.#client.messages.create({
        model: this.config.model,
        temperature: this.config.temperature,
        max_tokens: this.config.maxTokens,
        messages: [
          ...anthropicMessages,
          {
            role: "assistant",
            content: "I'll extract the structured data according to the schema.",
          },
        ],
        ...(system ? { system } : {}),
        tools: [extractionTool],
        tool_choice: { type: "tool", name: "extract_structured_data" },
      });

      // Extract the tool use from the response
      const toolUse = response.content.find(
        (block): block is ToolUseBlock => block.type === "tool_use",
      );

      if (!toolUse) {
        return { success: false, error: "No tool use found in response" };
      }

      // Parse and validate the result
      const result = params.schema.safeParse(toolUse.input);

      if (result.success) {
        return { success: true, data: result.data };
      } else {
        return { success: false, error: result.error.message };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Failed to generate object",
      };
    }
  }
}