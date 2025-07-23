import { defaults, defineAgent, definePlugin } from "life/define";
import z from "zod";

const tasksPlugin = definePlugin("tasks")
  .config(
    z.object({
      order: z.enum(["desc", "asc"]).default("desc"),
    }),
  )
  .events({
    "tasks.list": { dataSchema: z.array(z.string()) },
  })
  .api({
    schema: z.object({
      getTasks: z.function().args(z.string()).returns(z.array(z.string())),
    }),
    implementation: (Base, _schema) => {
      type Schema = z.infer<typeof _schema>;
      return class extends Base {
        getTasks: Schema["getTasks"] = (a) => [a];
      };
    },
  });

const collectionsPlugin = definePlugin("collections")
  .dependencies([
    defaults.plugins.core.pick({
      events: ["agent.resources-response", "messages.changed"],
      context: ["messages"],
    }),
  ])
  .config(
    z.object({
      collections: z.array(
        z.object({
          name: z.string(),
          description: z.string(),
        }),
      ),
    }),
  );

export default defineAgent("example")
  .plugins([...defaults.plugins, tasksPlugin, collectionsPlugin])
  .config({
    models: {
      llm: {
        provider: "mistral",
      },
      eou: {
        provider: "turnsense",
      },
    },
  });
