import "dotenv/config";
import { History } from "./agent/history";
import { AgentServer } from "./agent/server";
import { defaults, defineAgent, defineMemory } from "./exports/define";
import { getToken } from "./transport/auth";

async function main() {
  // Define the agent
  const builder = defineAgent("demo")
    .plugins([...defaults.plugins])
    .config({
      transport: {
        provider: "livekit",
      },
    })
    .core({})
    .memories({
      items: [
        defineMemory("instructions")
          .config({ behavior: "blocking" })
          .onHistoryChange(({ messages }) => {
            console.log("History changed:", messages);
          })
          .output(() => {
            const history = new History([]);
            history.createMessage({
              role: "system",
              content: "You are a helpful assistant called Lify.",
            });
            return history.getMessages();
          }),
        defineMemory("all-messages")
          .config({ behavior: "blocking" })
          .output(({ messages }) => messages),
      ],
    });

  // Instantiate the agent
  const agent = new AgentServer(builder._definition);

  // Handle graceful shutdown
  let isShuttingDown = false;
  const shutdown = async () => {
    if (isShuttingDown) return;
    isShuttingDown = true;

    console.log("\nReceived interrupt signal, shutting down gracefully...");
    await agent.stop();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  // Start the agent
  const roomId = "room-1";
  const token = await getToken("livekit", builder._definition.config.transport, roomId, agent.id);
  await agent.transport.joinRoom(roomId, token);
  await agent.start();
  console.log("Agent server started. Press Ctrl+C to stop.");

  // Keep the process alive
  await new Promise((resolve) => resolve(undefined));
}

main().catch((error) => {
  console.error("Error starting agent:", error);
  process.exit(1);
});
