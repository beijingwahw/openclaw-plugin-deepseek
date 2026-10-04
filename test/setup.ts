// Vitest setup: the sandbox environment may inject a DEEPSEEK_API_KEY that
// would leak into fail-closed tests. Clear it for deterministic plugin tests.
delete process.env.DEEPSEEK_API_KEY;
