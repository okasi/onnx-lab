import assert from 'node:assert/strict';
import test from 'node:test';
import { runBrowserHtml } from '../lib/browser-runtime.mjs';

test('browser runtime serves a page on an ephemeral port and reads its result', async (t) => {
  try {
    const { payload } = await runBrowserHtml(
      '<script>window.__RESULT__ = { status: "ok", value: 42 };</script>',
      { timeoutMs: 30_000 },
    );
    assert.deepEqual(payload, { status: 'ok', value: 42 });
  } catch (error) {
    if (/executable|chrome/i.test(error.message)) {
      t.skip(`Chrome unavailable: ${error.message}`);
      return;
    }
    throw error;
  }
});
