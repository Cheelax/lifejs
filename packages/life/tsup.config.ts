import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["exports/define.ts", "agent/client.ts", "transport/auth.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  treeshake: true,
});
