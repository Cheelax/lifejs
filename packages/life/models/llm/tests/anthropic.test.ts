import type { Message, ToolDefinition } from "@/agent/resources";
import { z } from "zod";
import { AnthropicLLM } from "../providers/anthropic";

function createMessage(role: Message["role"], content: string): Message {
  const now = Date.now();
  return {
    id: `msg-${Math.random().toString(36).substring(2, 11)}`,
    role,
    content,
    createdAt: now,
    lastUpdated: now,
  } as Message;
}

function createSearchTool(): ToolDefinition {
  return {
    name: "search",
    description: "Search for information on the web",
    inputSchema: z.object({
      query: z.string().describe("The search query"),
    }),
    outputSchema: z.object({
      result: z.string().describe("The search result"),
    }),
    run: async (input: object) => ({
      success: true,
      output: { result: `Mock search result for: ${(input as { query: string }).query}` },
    }),
  };
}

function createCalculatorTool(): ToolDefinition {
  return {
    name: "calculator",
    description: "Perform mathematical calculations",
    inputSchema: z.object({
      expression: z.string().describe("Mathematical expression to evaluate"),
    }),
    outputSchema: z.object({
      result: z.number().describe("The calculation result"),
    }),
    run: async () => ({ 
      success: true, 
      output: { result: Math.random() * 100 } 
    }),
  };
}

// Simple helper to consume a stream with timeout
async function consumeStream(
  job: {
    getStream: () => AsyncIterable<{ type: string; [key: string]: unknown }>;
    cancel: () => void;
  },
  timeoutMs = 10000,
) {
  const results = {
    content: "",
    tools: [] as Array<{ id: string; input: unknown }>,
    hasContent: false,
    toolsCalled: 0,
    error: null as string | null,
  };

  return new Promise<typeof results>((resolve) => {
    const timeout = setTimeout(() => {
      job.cancel();
      resolve(results);
    }, timeoutMs);

    let inactivityTimeout: NodeJS.Timeout | undefined;

    const checkInactivity = () => {
      if (inactivityTimeout) clearTimeout(inactivityTimeout);
      inactivityTimeout = setTimeout(() => {
        if (results.toolsCalled > 0 || results.hasContent) {
          clearTimeout(timeout);
          resolve(results);
        }
      }, 1000);
    };

    (async () => {
      try {
        for await (const chunk of job.getStream()) {
          if (inactivityTimeout) clearTimeout(inactivityTimeout);

          if (chunk.type === "content") {
            results.content += String(chunk.content);
            results.hasContent = true;
            checkInactivity();
          } else if (chunk.type === "tool") {
            results.tools.push({ id: String(chunk.toolId), input: chunk.toolInput });
            results.toolsCalled++;
            checkInactivity();
          } else if (chunk.type === "tools") {
            const tools = chunk.tools as Array<{ id: string; name: string; input: unknown }>;
            results.tools.push(...tools.map(tool => ({ id: tool.name, input: tool.input })));
            results.toolsCalled += tools.length;
            checkInactivity();
          } else if (chunk.type === "end") {
            break;
          } else if (chunk.type === "error") {
            results.error = String(chunk.error);
            break;
          }
        }
      } catch (error) {
        results.error = String(error);
      } finally {
        clearTimeout(timeout);
        if (inactivityTimeout) clearTimeout(inactivityTimeout);
        resolve(results);
      }
    })();
  });
}

async function runTests() {
  console.log("🚀 Testing Anthropic LLM Provider\n");

  if (!process.env.ANTHROPIC_API_KEY) {
    console.log("⚠️  Skipping Anthropic - ANTHROPIC_API_KEY not set");
    console.log("\nTo run tests, set the environment variable:");
    console.log("export ANTHROPIC_API_KEY=your_key_here");
    return false;
  }

  const provider = new AnthropicLLM({
    apiKey: process.env.ANTHROPIC_API_KEY,
    model: "claude-3-5-haiku-20241022",
    temperature: 0.1,
    maxTokens: 1024,
  });

  let passed = 0;
  let total = 0;

  // Test 1: Generate simple message
  total++;
  console.log("\n📝 Test 1: Generate Message");
  try {
    const messages = [
      createMessage("system", "Respond with exactly 'Hello World'"),
      createMessage("user", "Say hello"),
    ];

    const job = await provider.generateMessage({ messages, tools: [] });
    const result = await consumeStream(job, 8000);

    if (result.error) {
      console.log(`❌ Generate Message: ${result.error}`);
    } else if (result.content.length > 0) {
      console.log(`✅ Generate Message: "${result.content.trim()}"`);
      passed++;
    } else {
      console.log("❌ Generate Message: No response received");
    }
  } catch (error) {
    console.log(`❌ Generate Message: ${error}`);
  }

  // Test 2: Generate object
  total++;
  console.log("\n📦 Test 2: Generate Object");
  try {
    const messages = [
      createMessage("system", "Respond with valid JSON only"),
      createMessage("user", "Create a person with name 'John' and age 25"),
    ];

    const schema = z.object({
      name: z.string(),
      age: z.number(),
    });

    const result = await provider.generateObject({ messages, schema });

    if (result.success && result.data && typeof result.data === "object") {
      console.log(`✅ Generate Object: ${JSON.stringify(result.data)}`);
      passed++;
    } else {
      console.log(
        `❌ Generate Object: ${result.success ? "Invalid data structure" : (result as any).error}`,
      );
    }
  } catch (error) {
    console.log(`❌ Generate Object: ${error}`);
  }

  // Test 3: Single tool calling
  total++;
  console.log("\n🔧 Test 3: Single Tool Calling");
  try {
    const messages = [
      createMessage("system", "Use the search tool when asked to search. Be concise."),
      createMessage("user", "Search for TypeScript documentation"),
    ];

    const tools = [createSearchTool()];
    const job = await provider.generateMessage({ messages, tools });
    const result = await consumeStream(job, 15000);

    if (result.error) {
      console.log(`❌ Single Tool: ${result.error}`);
    } else if (result.toolsCalled > 0) {
      const toolDetails = result.tools.map((t) => `${t.id}(${JSON.stringify(t.input)})`).join(", ");
      console.log(`✅ Single Tool: ${result.toolsCalled} tool(s) called - ${toolDetails}`);
      console.log(`   Content generated: ${result.hasContent ? "Yes" : "No"}`);
      passed++;
    } else {
      console.log("❌ Single Tool: No tools called");
      console.log(`   Content generated: ${result.hasContent ? "Yes" : "No"}`);
    }
  } catch (error) {
    console.log(`❌ Single Tool: ${error}`);
  }

  console.log(
    `\n📊 Anthropic Results: ${passed}/${total} tests passed (${Math.round((passed / total) * 100)}%)`,
  );

  return passed === total;
}

export { runTests };

if (require.main === module) {
  runTests();
}