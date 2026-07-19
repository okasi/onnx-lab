import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  MemoryMonitor,
  classifyRuntimeError,
  readJson,
  runJsonWorker,
  summarizeTimings,
  writeJson,
} from '../lib/benchmark-support.mjs';

test('JSON helpers and worker runner cover stdout and result-file contracts', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'onnx-lab-test-'));
  const output = path.join(directory, 'value.json');
  await writeJson(output, { ok: true });
  assert.deepEqual(await readJson(output), { ok: true });

  const stdoutWorker = path.join(directory, 'stdout-worker.mjs');
  await fs.writeFile(stdoutWorker, 'console.log(JSON.stringify({status:"ok",value:3}));\n');
  const stdoutRun = await runJsonWorker(stdoutWorker, [], {
    exposeGc: false,
    resultFile: false,
  });
  assert.deepEqual(stdoutRun.result, { status: 'ok', value: 3 });

  const fileWorker = path.join(directory, 'file-worker.mjs');
  await fs.writeFile(
    fileWorker,
    `import fs from 'node:fs/promises';
const index = process.argv.indexOf('--result-file');
await fs.writeFile(process.argv[index + 1], JSON.stringify({status:'ok',value:4}));
`,
  );
  const fileRun = await runJsonWorker(fileWorker, [], {
    exposeGc: false,
    resultFile: true,
  });
  assert.deepEqual(fileRun.result, { status: 'ok', value: 4 });
});

test('timing, memory, and error helpers return stable shapes', async () => {
  assert.deepEqual(summarizeTimings([]), {
    count: 0,
    mean_ms: 0,
    total_ms: 0,
    p50_ms: 0,
    p95_ms: 0,
  });
  assert.deepEqual(summarizeTimings([1, 2, 10]), {
    count: 3,
    mean_ms: 4.33,
    total_ms: 13,
    p50_ms: 2,
    p95_ms: 2,
  });
  assert.equal(classifyRuntimeError('std::bad_alloc'), 'oom');
  assert.equal(classifyRuntimeError('requires f16 shader-f16'), 'shader_f16');

  const monitor = new MemoryMonitor(5);
  monitor.start();
  await new Promise((resolve) => setTimeout(resolve, 10));
  const peak = monitor.stop();
  assert.ok(peak.peak_rss_mb > 0);
});
