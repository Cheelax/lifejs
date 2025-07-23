export { defineAgent } from "@/agent/definition";
export { defineConfig } from "@/config/definition";
export { definePlugin } from "@/plugins/definition";
export { defineMemory } from "@/plugins/memories/definition";
export { defineStore } from "@/plugins/stores/definition";

import { corePlugin } from "@/plugins/core/server";
import { memoriesPlugin } from "@/plugins/memories/server";
import { storesPlugin } from "@/plugins/stores/server";

export const defaults = {
  plugins: {
    core: corePlugin,
    memories: memoriesPlugin,
    stores: storesPlugin,
    // Allows defaults to be used as an iterable, e.g., [...defaults.plugins]
    *[Symbol.iterator]() {
      for (const entry of Object.values(this)) yield entry;
    },
  },
} as const;
