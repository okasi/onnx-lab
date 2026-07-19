#!/usr/bin/env node
import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import {
  ROOT_DIR,
  projectPath,
  readJson,
} from '../lib/benchmark-support.mjs';
import { loadMultimodalTasks } from '../lib/gemma4-multimodal-suite.mjs';

const execFileAsync = promisify(execFile);

async function collectFiles(directory, extension) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectFiles(fullPath, extension));
    } else if (entry.name.endsWith(extension)) {
      files.push(fullPath);
    }
  }
  return files;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function uniqueIds(items, label) {
  const ids = items.map((item) => item.id);
  assert(ids.every(Boolean), `${label} contains an item without an id`);
  assert(new Set(ids).size === ids.length, `${label} contains duplicate ids`);
}

async function checkJavaScript() {
  const directories = ['config', 'lib', 'scripts', 'test']
    .map((name) => projectPath(name));
  const files = (await Promise.all(
    directories.map((directory) => collectFiles(directory, '.mjs').catch(() => [])),
  )).flat();
  await Promise.all(
    files.map((file) => execFileAsync(process.execPath, ['--check', file], { cwd: ROOT_DIR })),
  );

  const importable = files.filter((file) =>
    file.includes(`${path.sep}config${path.sep}`)
    || file.includes(`${path.sep}lib${path.sep}`));
  await Promise.all(importable.map((file) => import(file)));
  return files.length;
}

async function checkCorpus() {
  const corpus = await readJson(projectPath('data', 'benchmark-corpus.json'));
  uniqueIds(corpus.documents, 'benchmark corpus');
  uniqueIds(corpus.query_pairs, 'benchmark query pairs');
  assert(corpus.documents.length >= 50, 'benchmark corpus must contain at least 50 documents');
  const ids = new Set(corpus.documents.map((document) => document.id));
  for (const document of corpus.documents) {
    assert(document.char_count === document.text.length, `${document.id} char_count is stale`);
    assert(
      document.word_count === document.text.split(/\s+/).length,
      `${document.id} word_count is stale`,
    );
  }
  for (const pair of corpus.query_pairs) {
    assert(ids.has(pair.sv_doc_id), `${pair.id} references missing Swedish document`);
    assert(ids.has(pair.tr_doc_id), `${pair.id} references missing Turkish document`);
  }
  assert(corpus.stats.document_count === corpus.documents.length, 'corpus document stats are stale');
  assert(corpus.stats.query_pair_count === corpus.query_pairs.length, 'corpus pair stats are stale');
}

async function checkSuites() {
  const prompts = await readJson(projectPath('data', 'gemma4-benchmark-prompts.json'));
  uniqueIds(prompts.prompts, 'Gemma 4 benchmark prompts');

  const multimodal = await readJson(projectPath('data', 'gemma4-multimodal-suite.json'));
  const multimodalTasks = loadMultimodalTasks(multimodal);
  uniqueIds(multimodalTasks, 'multimodal suite');
  for (const task of multimodalTasks) {
    assert(['image', 'audio'].includes(task.modality), `${task.id} has invalid modality`);
    assert(task.media_url, `${task.id} has no media URL`);
  }

  const quality = await readJson(projectPath('data', 'gemma4-quality-suite.json'));
  const categories = Object.entries(quality.categories);
  const totalWeight = categories.reduce((sum, [, category]) => sum + category.weight, 0);
  assert(Math.abs(totalWeight - 1) < 1e-9, 'quality category weights must sum to 1');
  const allTasks = categories.flatMap(([, category]) => category.tasks);
  uniqueIds(allTasks, 'quality suite');
}

async function checkPackageScripts() {
  const pkg = await readJson(projectPath('package.json'));
  for (const [name, command] of Object.entries(pkg.scripts)) {
    const matches = command.matchAll(
      /\b(?:node|bash)\s+(?:--expose-gc\s+)?(scripts\/[^\s&]+)/g,
    );
    for (const match of matches) {
      const script = projectPath(match[1]);
      assert(
        await fs.stat(script).catch(() => null),
        `package script ${name} references ${match[1]}`,
      );
    }
  }
}

const fileCount = await checkJavaScript();
await checkCorpus();
await checkSuites();
await checkPackageScripts();
console.log(`Checks passed: ${fileCount} JavaScript files, package scripts, and all data suites`);
