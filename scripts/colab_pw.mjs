// Playwriter backend for colab_sync.py: reads an async page function from stdin,
// runs it in the Colab tab for argv[2] (Drive id), opening that notebook in a
// background tab when no attached tab has it. Prints one line: PWJSON {...}.
// Drive id "__ANY__" (account / kill-switch calls that need no notebook): reuse any open
// colab.research.google.com tab untouched (never reloaded or navigated); with none open, a
// temporary tab on https://colab.research.google.com/ is used and closed right after evaluate.
// pwError means Playwriter itself is unavailable and the caller should fall back.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [driveId, url] = process.argv.slice(2);
const js = fs.readFileSync(0, "utf8");
const out = (o) => process.stdout.write("PWJSON " + JSON.stringify(o) + "\n");
const firstLine = (e) => String(e?.message || e).split("\n")[0];
// PLAYWRITER_ENTRY overrides; the default is the Windows global npm prefix (`npm install -g playwriter`).
const entry = process.env.PLAYWRITER_ENTRY
  || path.join(process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming"), "npm", "node_modules", "playwriter", "dist", "index.js");
process.env.PLAYWRITER_TELEMETRY ??= "0";

let conn;
try {
  if (!fs.existsSync(entry)) throw new Error("playwriter is not installed (npm install -g playwriter, or set PLAYWRITER_ENTRY to its dist/index.js)");
  const { connectViaExtension } = await import(pathToFileURL(entry).href);
  conn = await Promise.race([
    connectViaExtension({ tabGroup: "colab", tabGroupColor: "orange" }),
    new Promise((_, reject) => setTimeout(() => reject(new Error("connect timeout")), 15000)),
  ]);
} catch (e) {
  out({ pwError: firstLine(e) });
  process.exit(0);
}
try {
  const context = conn.browser.contexts()[0];
  const anyTab = driveId === "__ANY__";
  let page = anyTab
    ? context.pages().findLast((p) => /^https:\/\/colab\.research\.google\.com(\/|$)/.test(p.url()))
    : context.pages().findLast((p) => p.url().includes(driveId));
  let temporary = false;
  if (!page) {
    page = await context.newPage();
    temporary = anyTab;
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
  }
  try {
    out({ result: await page.evaluate(`(${js})()`) });
  } catch (e) {
    out({ result: { error: "evaluate failed", detail: firstLine(e) } });
  } finally {
    if (temporary) await Promise.race([page.close().catch(() => {}), new Promise((r) => setTimeout(r, 5000))]);
  }
} catch (e) {
  out({ pwError: firstLine(e) });
} finally {
  await Promise.race([conn.close(), new Promise((r) => setTimeout(r, 5000))]);
  process.exit(0);
}
