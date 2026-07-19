import http from 'node:http';

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return server.address().port;
}

async function close(server) {
  if (!server.listening) {
    return;
  }
  await new Promise((resolve) => server.close(resolve));
}

export async function runBrowserHtml(
  html,
  {
    timeoutMs = 600_000,
    launchOptions = {},
    logElementId = null,
  } = {},
) {
  const server = http.createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    response.end(html);
  });
  const port = await listen(server);

  let browser;
  const startedAt = performance.now();
  try {
    const { chromium } = await import('playwright-core');
    browser = await chromium.launch({
      channel: 'chrome',
      headless: true,
      ...launchOptions,
    });
    const page = await browser.newPage();
    page.setDefaultTimeout(timeoutMs);
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__RESULT__ !== undefined, null, {
      timeout: timeoutMs,
    });
    const payload = await page.evaluate((elementId) => {
      const result = { ...window.__RESULT__ };
      if (elementId) {
        result.log = document.getElementById(elementId)?.textContent ?? '';
      }
      return result;
    }, logElementId);

    return {
      payload,
      total_ms: performance.now() - startedAt,
    };
  } finally {
    await browser?.close().catch(() => {});
    await close(server);
  }
}
