import { defineConfig, devices } from "@playwright/test";
import { resolve } from "node:path";

const root = resolve(__dirname, "../../..");
const baseURL = "http://127.0.0.1:5199";

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.ts",
  outputDir: resolve(root, "src/studio/frontend/node_modules/.cache/studio-e2e"),
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  timeout: 30_000,
  use: { ...devices["Desktop Chrome"], baseURL, trace: "retain-on-failure" },
  webServer: {
    command: "npm --prefix src/studio/frontend run build && uv run python tests/studio/e2e/serve_fixture.py",
    cwd: root,
    url: `${baseURL}/api/health`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
