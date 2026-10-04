import { expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

test("manifest declares provider, harness, and tools contracts", () => {
  const manifest = JSON.parse(readFileSync(join(root, "openclaw.plugin.json"), "utf8"));
  expect(manifest.id).toBe("openclaw-deepseek");
  expect(manifest.providers).toContain("openclaw-deepseek");
  expect(manifest.providerAuthAliases["openclaw-deepseek"]).toBe("deepseek");
  expect(manifest.activation.onAgentHarnesses).toContain("openclaw-deepseek");
  expect(manifest.contracts.tools).toEqual(
    expect.arrayContaining(["deepseek_research", "deepseek_reason", "deepseek_balance"]),
  );
  for (const tool of manifest.contracts.tools) {
    expect(manifest.toolMetadata[tool]).toBeDefined();
  }
  expect(manifest.configSchema.additionalProperties).toBe(false);
});

test("package.json points runtime entries at built JS", () => {
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  expect(pkg.openclaw.extensions).toContain("./dist/index.js");
  expect(pkg.openclaw.runtimeExtensions).toContain("./dist/index.js");
  expect(pkg.dependencies.typebox).toBeDefined();
  expect(pkg.peerDependencies.openclaw).toBeDefined();
});
