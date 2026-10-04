#!/usr/bin/env node
/**
 * Structural validation for the openclaw-deepseek plugin (no gateway needed).
 * Checks manifest shape, package metadata, and built entry exports.
 */
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const errors = [];

function check(condition, message) {
  if (!condition) errors.push(message);
}

const manifestPath = join(root, "openclaw.plugin.json");
check(existsSync(manifestPath), "openclaw.plugin.json is missing");
if (existsSync(manifestPath)) {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  check(manifest.id === "openclaw-deepseek", "manifest.id must be openclaw-deepseek");
  check(Array.isArray(manifest.providers) && manifest.providers.includes("openclaw-deepseek"), "manifest.providers must include openclaw-deepseek");
  check(manifest.providerAuthAliases?.["openclaw-deepseek"] === "deepseek", "providerAuthAliases must map openclaw-deepseek -> deepseek");
  check(Array.isArray(manifest.contracts?.tools), "contracts.tools must be an array");
  check(manifest.configSchema?.type === "object", "configSchema must be an object schema");
  check(manifest.activation?.onAgentHarnesses?.includes("openclaw-deepseek"), "activation.onAgentHarnesses must include openclaw-deepseek");
}

const pkgPath = join(root, "package.json");
const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
check(pkg.type === "module", "package must be ESM");
check(pkg.openclaw?.extensions?.includes("./dist/index.js"), "openclaw.extensions must include ./dist/index.js");
check(pkg.openclaw?.runtimeExtensions?.includes("./dist/index.js"), "openclaw.runtimeExtensions must include ./dist/index.js");
check(pkg.dependencies?.typebox, "typebox must be a runtime dependency");

const distEntry = join(root, "dist", "index.js");
check(existsSync(distEntry), "dist/index.js is missing; run npm run build first");
if (existsSync(distEntry)) {
  const { default: entry } = await import(distEntry);
  check(entry && typeof entry === "object", "dist entry must default-export a plugin entry");
  check(typeof entry.register === "function", "plugin entry must expose register()");
}

if (errors.length > 0) {
  console.error("openclaw-deepseek plugin validation FAILED:");
  for (const error of errors) console.error(` - ${error}`);
  process.exit(1);
}
console.log("openclaw-deepseek plugin is valid.");
