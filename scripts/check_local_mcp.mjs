import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir, homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { crc32, deflateSync } from 'node:zlib';

function pngChunk(type, data) {
  const body = Buffer.concat([Buffer.from(type), data]);
  const prefix = Buffer.alloc(4), checksum = Buffer.alloc(4);
  prefix.writeUInt32BE(data.length);
  checksum.writeUInt32BE(crc32(body));
  return Buffer.concat([prefix, body, checksum]);
}
function pixelPNG() {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(1,0); header.writeUInt32BE(1,4); header[8]=8; header[9]=6;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), pngChunk('IHDR',header),
    pngChunk('IDAT',deflateSync(Buffer.from([0,255,0,0,255]))), pngChunk('IEND',Buffer.alloc(0))]);
}

function findPackageRoot() {
  if (process.env.LOCAL_MCP_PI_ROOT) return process.env.LOCAL_MCP_PI_ROOT;
  if (process.platform === 'win32') {
    return join(process.env.APPDATA || join(homedir(), 'AppData/Roaming'), 'npm/node_modules/@earendil-works/pi-coding-agent');
  }
  const prefixes = [
    process.env.npm_config_prefix,
    join(process.execPath, '..', '..'),
    '/opt/homebrew',
    '/usr/local',
    '/usr',
    join(homedir(), '.npm-global'),
  ].filter(Boolean);
  for (const prefix of prefixes) {
    const cand = join(prefix, 'lib/node_modules/@earendil-works/pi-coding-agent');
    if (existsSync(join(cand, 'package.json'))) return cand;
  }
  return join(process.env.npm_config_prefix || '/usr/local', 'lib/node_modules/@earendil-works/pi-coding-agent');
}
const packageRoot = findPackageRoot();
const { McpClient, StreamableHttpTransport } = await import(pathToFileURL(join(packageRoot, 'node_modules/@earendil-works/pi-mcp/dist/index.js')).href);
const launcher = process.argv[2];
assert(launcher, 'Pass the absolute local_mcp.py path');
const dir = mkdtempSync(join(tmpdir(), 'local-mcp-client-check-'));
const config = join(dir, 'config.json');
const token = randomBytes(32).toString('base64url');
writeFileSync(config, JSON.stringify({ token, cwd: dir }));
const port = await new Promise(resolve => {
  const s = createServer();
  s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => resolve(p)); });
});
const env = { ...process.env, LOCAL_MCP_CONFIG: config, LOCAL_MCP_PORT: String(port), LOCAL_MCP_LOG: join(dir, 'server.log') };
const pythonBin = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
function lifecycle(command) {
  const p = spawnSync(pythonBin, ['-X', 'utf8', launcher, command], { env, encoding: 'utf8', timeout: 16000, windowsHide: true });
  const lines = (p.stdout || '').trim().split('\n');
  let result;
  try { result = JSON.parse(lines.at(-1)); } catch { throw new Error(`${command}: no JSON result (${p.error?.message || p.stderr})`); }
  assert.equal(result.ok, true, `${command}: ${result.error || p.stderr}`);
  return result;
}
const client = new McpClient({ name: 'release-check', version: '1' });
let started = false;
try {
  const first = lifecycle('start');
  started = true;
  const second = lifecycle('start');
  assert.equal(first.pid, second.pid, 'Repeated launch must reuse one process');
  const base = `http://127.0.0.1:${port}/mcp`;
  const rejected = await fetch(base, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ jsonrpc:'2.0', id:1, method:'tools/list' }) });
  assert.equal(rejected.status, 401);
  const wrong = await fetch(base, {method:'POST',headers:{Authorization:'Bearer wrong','Content-Type':'application/json'},body:'{}'});
  assert.equal(wrong.status, 401);
  const publicHealth = await (await fetch(base.replace('/mcp','/health'))).json();
  assert.equal(publicHealth.server, 'local_mcp');
  for (const key of ['token','cwd','pid']) assert(!(key in publicHealth));
  await client.connect(new StreamableHttpTransport({ url: base, headers: { Authorization: `Bearer ${token}` } }));
  const tools = await client.listTools();
  assert.deepEqual(tools.map(t=>t.name).sort(), ['bash','edit','read','write']);
  const written = await client.callTool('write', {path:'fixture.txt',content:'alpha\nbeta\n'});
  assert(!written.isError);
  const edited = await client.callTool('edit', {path:'fixture.txt',edits:[{oldText:'beta',newText:'gamma'}]});
  assert(!edited.isError);
  const read = await client.callTool('read', {path:'fixture.txt',offset:2,limit:1});
  assert(read.content.some(c=>c.type==='text' && c.text.includes('gamma')));
  const loose = await client.callTool('bash', {command:'printf native-coercion-ok',timeout:'5'});
  assert(loose.content.some(c=>c.type==='text' && c.text.includes('native-coercion-ok')));
  const nullable = await client.callTool('read', {path:'fixture.txt',offset:null,limit:'1'});
  assert(!nullable.isError && nullable.content.some(c=>c.type==='text' && c.text.includes('alpha')));
  const bash = await client.callTool('bash', {command:'printf mcp-native-ok',timeout:5});
  assert(bash.content.some(c=>c.type==='text' && c.text.includes('mcp-native-ok')));
  const failed = await client.callTool('bash', {command:'exit 7',timeout:5});
  assert.equal(failed.isError, true);
  const absent = await client.callTool('read', {path:'does-not-exist.txt'});
  assert.equal(absent.isError, true);
  const timed = await client.callTool('bash', {command:'sleep 3',timeout:1});
  assert.equal(timed.isError, true);
  assert(timed.content.some(c=>c.type==='text' && /timed out|timeout/i.test(c.text)));
  writeFileSync(join(dir,'large.txt'), Array.from({length:10000},(_,i)=>`line-${i}`).join('\n'));
  const large = await client.callTool('read', {path:'large.txt'});
  assert(large.content.some(c=>c.type==='text' && /offset|truncat/i.test(c.text)));
  writeFileSync(join(dir,'pixel.png'), pixelPNG());
  const image = await client.callTool('read', {path:'pixel.png'});
  assert(image.content.some(c=>c.type==='image' && c.mimeType==='image/png' && c.data), JSON.stringify(image));
  await client.callTool('bash', {command:`printf ${token}; exit 1`,timeout:5});
  if (existsSync(join(dir,'server.log'))) assert(!readFileSync(join(dir,'server.log'),'utf8').includes(token));
  for (const [name,result] of [['write',written],['edit',edited],['read',read],['bash',bash]]) {
    if (result.structuredContent !== undefined) assert(result.structuredContent && typeof result.structuredContent==='object' && !Array.isArray(result.structuredContent));
    const schema = tools.find(t=>t.name===name).outputSchema;
    if (schema) {
      assert.equal(schema.type, 'object', `${name}: MCP outputSchema must describe an object`);
      assert(result.structuredContent !== undefined, `${name}: declared outputSchema requires matching structuredContent`);
    }
  }
  console.log(JSON.stringify({ok:true,client:'Pi McpClient',tools:tools.map(t=>t.name),checks:['401 and wrong token','private health metadata','single instance','write','exact edit','partial read','native argument coercion','bash','nonzero exit','missing file','timeout','truncation','image','token-free logs','MCP result contract']}));
} finally {
  await client.close().catch(()=>{});
  if (started) lifecycle('stop');
  rmSync(dir, {recursive:true,force:true});
}
