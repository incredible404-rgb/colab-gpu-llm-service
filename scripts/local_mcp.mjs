#!/usr/bin/env node
/**
 * local_mcp: Streamable-HTTP MCP server that exposes this machine's Pi-native read/write/edit/bash tools.
 *
 * The four tools are the installed Pi coding agent's own implementations (createCodingTools from
 * @earendil-works/pi-coding-agent), imported from the global npm package at startup, so image reads,
 * truncation notices, exact multi-edit and bash timeouts behave exactly as they do inside Pi. Nothing is
 * re-implemented here and the Pi installation is never modified.
 *
 * Every POST /mcp needs `Authorization: Bearer <token>`; the token and the default working directory are
 * read from the JSON file local_mcp.py writes (LOCAL_MCP_CONFIG redirects it for isolated tests). GET
 * /health is unauthenticated and only names the server, so a launcher can tell this bridge apart from the
 * legacy local_novel_mcp.py on the same port without leaking a token, a pid or a path.
 *
 * Usage: node local_mcp.mjs [--print-tools]   (normal operation is through `local_mcp.py start`)
 */
import { timingSafeEqual } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { isIPv4 } from 'node:net';
import { homedir } from 'node:os';
import { delimiter, dirname, isAbsolute, join, resolve as resolvePath } from 'node:path';
import { pathToFileURL } from 'node:url';

const SERVER_NAME = 'local_mcp';
const SERVER_VERSION = '1.0.0';
const DEFAULT_PORT = 3426;
const DEFAULT_HOST = '127.0.0.1';
// A tools/call argument carries a file body (write) or an edit batch, so the default 1 MiB is too small.
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const SUPPORTED_PROTOCOLS = ['2025-06-18', '2025-03-26', '2024-11-05'];
const MAX_SESSIONS = 64;
const PI_PACKAGE = '@earendil-works/pi-coding-agent';

const configPath = () =>
	process.env.LOCAL_MCP_CONFIG || join(homedir(), '.config', 'colab-cli', 'local-mcp.json');

/**
 * Write one log line to stderr. The launcher redirects stderr into ~/.config/colab-cli/local-mcp.log, so the
 * server is the only writer of that file. Never pass a token here.
 */
let privateToken = '';

function logLine(text) {
	const safe = privateToken ? String(text).replaceAll(privateToken, '[redacted]') : String(text);
	console.error(`[local_mcp] ${new Date().toISOString()} ${safe}`);
}

function fatal(message, code = 1) {
	logLine(`fatal: ${message}`);
	process.exit(code);
}

function readConfig() {
	const path = configPath();
	let text;
	try {
		text = readFileSync(path, 'utf8');
	} catch (err) {
		fatal(`cannot read ${path} (${err.code || err.message}); run: python -X utf8 local_mcp.py start`);
	}
	let data;
	try {
		data = JSON.parse(text);
	} catch (err) {
		fatal(`${path} is not valid JSON`);
	}
	if (!data || typeof data !== 'object' || Array.isArray(data)) fatal(`${path} must hold a JSON object`);
	return data;
}

/**
 * Directory of the globally installed Pi coding agent.
 *
 * Resolved from the platform's standard npm global root (or LOCAL_MCP_PI_ROOT), never from a hard-coded
 * absolute path, so the same file works on another account or machine.
 */
function resolvePiDir() {
	const candidates = [];
	const explicit = (process.env.LOCAL_MCP_PI_ROOT || '').trim();
	if (explicit) candidates.push(explicit);
	const roots = [];
	if (process.platform === 'win32') {
		const appData = process.env.APPDATA || join(homedir(), 'AppData', 'Roaming');
		roots.push(join(appData, 'npm', 'node_modules'));
		roots.push(join(dirname(process.execPath), 'node_modules'));
	} else {
		if (process.env.npm_config_prefix) roots.push(join(process.env.npm_config_prefix, 'lib', 'node_modules'));
		roots.push(
			join(dirname(dirname(process.execPath)), 'lib', 'node_modules'),
			'/opt/homebrew/lib/node_modules',
			'/usr/local/lib/node_modules',
			'/usr/lib/node_modules',
			join(homedir(), '.npm-global', 'lib', 'node_modules'),
		);
	}
	for (const entry of (process.env.NODE_PATH || '').split(delimiter)) {
		if (entry) roots.push(entry);
	}
	for (const root of roots) candidates.push(join(root, ...PI_PACKAGE.split('/')));
	for (const dir of candidates) {
		if (existsSync(join(dir, 'package.json'))) return dir;
	}
	fatal(
		`${PI_PACKAGE} not found (looked in: ${candidates.join(', ')}); install it with "npm i -g ${PI_PACKAGE}" ` +
			'or point LOCAL_MCP_PI_ROOT at the package directory',
	);
}

/** Import createCodingTools and the schema validator from one Pi installation. */
async function loadPi(piDir) {
	let pkg;
	try {
		pkg = JSON.parse(readFileSync(join(piDir, 'package.json'), 'utf8'));
	} catch (err) {
		fatal(`cannot read ${join(piDir, 'package.json')} (${err.code || err.message})`);
	}
	const exportEntry = pkg.exports && pkg.exports['.'] && (pkg.exports['.'].import || pkg.exports['.'].default);
	const main = typeof exportEntry === 'string' ? exportEntry : pkg.main || './dist/index.js';
	const entry = resolvePath(piDir, main);
	if (!existsSync(entry)) fatal(`Pi entry ${entry} is missing; the installed package looks broken`);
	let mod;
	try {
		mod = await import(pathToFileURL(entry).href);
	} catch (err) {
		fatal(`cannot import ${entry} (${err.message})`);
	}
	if (typeof mod.createCodingTools !== 'function') {
		fatal(`${entry} does not export createCodingTools; this ${PI_PACKAGE} version is not supported`);
	}
	const require = createRequire(join(piDir, 'package.json'));
	let validateToolArguments;
	try {
		const aiDir = (require.resolve.paths('@earendil-works/pi-ai') || [])
			.map((root) => join(root, '@earendil-works', 'pi-ai'))
			.find((dir) => existsSync(join(dir, 'package.json')));
		if (!aiDir) throw new Error('@earendil-works/pi-ai was not found in Pi dependencies');
		const aiPackage = JSON.parse(readFileSync(join(aiDir, 'package.json'), 'utf8'));
		const aiEntry = aiPackage.exports?.['.']?.import || aiPackage.main;
		({ validateToolArguments } = await import(pathToFileURL(resolvePath(aiDir, aiEntry)).href));
	} catch (err) {
		fatal(`cannot load Pi's argument validator from ${piDir} (${err.message})`);
	}
	if (typeof validateToolArguments !== 'function') fatal('Pi does not export validateToolArguments');
	return { createCodingTools: mod.createCodingTools, validateToolArguments, version: pkg.version };
}

function buildTools(createCodingTools, cwd) {
	let tools;
	try {
		tools = createCodingTools(cwd);
	} catch (err) {
		fatal(`createCodingTools(${cwd}) failed: ${err.message}`);
	}
	const byName = new Map();
	for (const tool of Array.isArray(tools) ? tools : []) {
		if (tool && typeof tool.name === 'string' && typeof tool.execute === 'function') byName.set(tool.name, tool);
	}
	for (const name of ['read', 'write', 'edit', 'bash']) {
		if (!byName.has(name)) fatal(`Pi did not provide the "${name}" tool; refusing to serve a partial tool set`);
	}
	return byName;
}

function toolDescriptors(tools) {
	return [...tools.values()].map((tool) => ({
		name: tool.name,
		description: typeof tool.description === 'string' ? tool.description : tool.name,
		inputSchema: tool.parameters,
	}));
}

class RpcError extends Error {
	constructor(code, message) {
		super(message);
		this.code = code;
	}
}

function rpcError(id, code, message) {
	return { jsonrpc: '2.0', id: id === undefined ? null : id, error: { code, message } };
}

/** Compare the bearer token in constant time; a wrong or missing header is a plain false. */
function isAuthorized(header, token) {
	const match = /^Bearer\s+(.+)$/i.exec(typeof header === 'string' ? header.trim() : '');
	if (!match) return false;
	const given = Buffer.from(match[1].trim(), 'utf8');
	const expected = Buffer.from(token, 'utf8');
	return given.length === expected.length && timingSafeEqual(given, expected);
}

function corsHeaders() {
	return {
		'Access-Control-Allow-Origin': '*',
		'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type, Accept, Authorization, MCP-Protocol-Version, Mcp-Session-Id',
		'Access-Control-Expose-Headers': 'Mcp-Session-Id, MCP-Protocol-Version',
		'Access-Control-Allow-Private-Network': 'true',
	};
}

function send(res, status, payload, extraHeaders = {}) {
	const body = payload === undefined ? '' : JSON.stringify(payload);
	const headers = {
		...corsHeaders(),
		...extraHeaders,
		...(body ? { 'Content-Type': 'application/json; charset=utf-8' } : {}),
		'Content-Length': Buffer.byteLength(body),
	};
	res.writeHead(status, headers);
	res.end(body);
}

/** Read the request body, draining but discarding anything past the limit so the connection stays usable. */
function readBody(req) {
	return new Promise((resolve, reject) => {
		let size = 0;
		let tooLarge = false;
		const chunks = [];
		req.on('data', (chunk) => {
			size += chunk.length;
			if (size > MAX_BODY_BYTES) {
				tooLarge = true;
				chunks.length = 0;
			} else {
				chunks.push(chunk);
			}
		});
		req.on('end', () => {
			if (tooLarge) reject(new RpcError(-32600, `request body exceeds ${MAX_BODY_BYTES} bytes`));
			else resolve(Buffer.concat(chunks).toString('utf8'));
		});
		req.on('error', reject);
	});
}

function main() {
	const args = process.argv.slice(2);
	if (args.includes('--help') || args.includes('-h')) {
		console.log('usage: node local_mcp.mjs [--print-tools]  (start it with: python -X utf8 local_mcp.py start)');
		return;
	}
	const printTools = args.includes('--print-tools');
	const config = readConfig();
	const token = typeof config.token === 'string' ? config.token.trim() : '';
	privateToken = token;
	if (!token) fatal(`${configPath()} has no token; run: python -X utf8 local_mcp.py start`);

	const rawCwd = typeof config.cwd === 'string' && config.cwd.trim() ? config.cwd.trim() : process.cwd();
	const cwd = isAbsolute(rawCwd) ? rawCwd : resolvePath(rawCwd);
	let cwdIsDir = false;
	try {
		cwdIsDir = statSync(cwd).isDirectory();
	} catch {
		cwdIsDir = false;
	}

	const piDir = resolvePiDir();
	loadPi(piDir)
		.then(({ createCodingTools, validateToolArguments, version }) => {
			const tools = buildTools(createCodingTools, cwd);
			if (printTools) {
				console.log(
					JSON.stringify({
						ok: true,
						pi: piDir,
						pi_version: version,
						cwd,
						cwd_exists: cwdIsDir,
						tools: [...tools.keys()],
					}),
				);
				return;
			}
			if (!cwdIsDir) fatal(`working directory ${cwd} does not exist; fix "cwd" in ${configPath()}`);
			serve({ token, cwd, tools, validateToolArguments, piVersion: version });
		})
		.catch((err) => fatal(`startup failed: ${err && err.message}`));
}

function serve({ token, cwd, tools, validateToolArguments, piVersion }) {
	const startedAt = new Date().toISOString();
	const descriptors = toolDescriptors(tools);
	const sessions = new Map();
	const port = Number(process.env.LOCAL_MCP_PORT || DEFAULT_PORT);
	// An explicit --host / LOCAL_MCP_HOST binds exactly that. Otherwise 127.0.0.1, plus the Tailscale IP the
	// launcher passes as LOCAL_MCP_EXTRA_HOST (or LOCAL_TAILSCALE_IP when run directly) so the cloud tunnel
	// reaches it without exposing the bridge on the LAN.
	const hostFlag = process.argv.indexOf('--host');
	const explicitHost = (hostFlag > 1 ? process.argv[hostFlag + 1] : '') || process.env.LOCAL_MCP_HOST || '';
	const host = explicitHost || DEFAULT_HOST;
	const extraCandidate = explicitHost
		? ''
		: (process.env.LOCAL_MCP_EXTRA_HOST ?? process.env.LOCAL_TAILSCALE_IP ?? '').trim();
	const extraHost =
		isIPv4(extraCandidate) && !['127.0.0.1', '0.0.0.0', host].includes(extraCandidate) ? extraCandidate : '';
	if (!Number.isInteger(port) || port <= 0 || port > 65535) fatal(`invalid LOCAL_MCP_PORT ${process.env.LOCAL_MCP_PORT}`);

	const instructions =
		`本机 MCP 桥（${SERVER_NAME}）：read/write/edit/bash 直接在本机执行，默认工作目录 ${cwd}。` +
		'工具接受绝对路径，bash 命令以本机当前登录用户的完整权限运行（不是沙箱）。' +
		'read 返回图片附件与截断提示，edit 一次可提交多处精确替换，bash 可传 timeout 秒数。';

	function toToolResult(result) {
		const content = [];
		for (const item of result && Array.isArray(result.content) ? result.content : []) {
			if (!item || typeof item !== 'object') continue;
			if (item.type === 'text' && typeof item.text === 'string') {
				content.push({ type: 'text', text: item.text });
			} else if (item.type === 'image' && typeof item.data === 'string' && typeof item.mimeType === 'string') {
				content.push({ type: 'image', data: item.data, mimeType: item.mimeType });
			}
		}
		// MCP requires a non-empty content array, and Pi's write/edit return text only on failure.
		if (content.length === 0) content.push({ type: 'text', text: '(no output)' });
		const out = { content };
		if (result && result.isError) out.isError = true;
		return out;
	}

	async function callTool(params, controller) {
		const name = params && params.name;
		if (typeof name !== 'string' || !name) throw new RpcError(-32602, 'Invalid params: name must be a non-empty string');
		const tool = tools.get(name);
		if (!tool) {
			throw new RpcError(-32602, `Invalid params: unknown tool "${name}" (available: ${[...tools.keys()].join(', ')})`);
		}
		let args = params.arguments === undefined || params.arguments === null ? {} : params.arguments;
		if (typeof args === 'string') {
			// Some clients stringify arguments; Pi's own runtime accepts the prepared form, so parse first.
			try {
				args = JSON.parse(args);
			} catch {
				throw new RpcError(-32602, 'Invalid params: arguments is a string but not valid JSON');
			}
		}
		if (typeof args !== 'object' || Array.isArray(args)) {
			throw new RpcError(-32602, 'Invalid params: arguments must be an object');
		}
		if (typeof tool.prepareArguments === 'function') {
			try {
				args = tool.prepareArguments(args);
			} catch (err) {
				throw new RpcError(-32602, `Invalid params for ${name}: ${err.message}`);
			}
		}
		try {
			args = validateToolArguments(tool, { name, arguments: args });
		} catch (err) {
			return { content: [{ type: 'text', text: err.message }], isError: true };
		}

		const toolCallId = `${SERVER_NAME}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
		const started = Date.now();
		try {
			const result = await tool.execute(toolCallId, args, controller.signal);
			logLine(`tools/call ${name} ok in ${Date.now() - started}ms`);
			return toToolResult(result);
		} catch (err) {
			// Pi reports a failed command by throwing (timeout, abort, missing path): the model must see it as
			// a tool error result, not as a broken JSON-RPC call.
			const message = err instanceof Error ? err.message : String(err);
			// Tool failures can contain file text or command output; keep that in the authenticated response.
			logLine(`tools/call ${name} failed in ${Date.now() - started}ms`);
			return { content: [{ type: 'text', text: message || 'tool failed' }], isError: true };
		}
	}

	async function dispatch(message, state) {
		if (!message || typeof message !== 'object' || Array.isArray(message)) {
			return rpcError(null, -32600, 'Invalid Request: expected a JSON-RPC object');
		}
		const hasId = Object.prototype.hasOwnProperty.call(message, 'id');
		// Every notification is ignored: this server sends no server-initiated messages.
		if (!hasId) return null;
		const id = message.id;
		if (message.jsonrpc !== '2.0') return rpcError(id, -32600, 'Invalid Request: jsonrpc must be "2.0"');
		if (typeof message.method !== 'string' || !message.method) {
			return rpcError(id, -32600, 'Invalid Request: method must be a non-empty string');
		}
		const params = message.params && typeof message.params === 'object' ? message.params : {};
		try {
			switch (message.method) {
				case 'initialize': {
					const requested = typeof params.protocolVersion === 'string' ? params.protocolVersion : '';
					state.protocolVersion = SUPPORTED_PROTOCOLS.includes(requested) ? requested : SUPPORTED_PROTOCOLS[0];
					return {
						jsonrpc: '2.0',
						id,
						result: {
							protocolVersion: state.protocolVersion,
							capabilities: { tools: { listChanged: false } },
							serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
							instructions,
						},
					};
				}
				case 'ping':
					return { jsonrpc: '2.0', id, result: {} };
				case 'tools/list':
					return { jsonrpc: '2.0', id, result: { tools: descriptors } };
				case 'tools/call':
					return { jsonrpc: '2.0', id, result: await callTool(params, state.controller) };
				default:
					throw new RpcError(-32601, `Method not found: ${message.method}`);
			}
		} catch (err) {
			if (err instanceof RpcError) return rpcError(id, err.code, err.message);
			logLine(`internal error in ${message.method}: ${err && err.message}`);
			return rpcError(id, -32603, `Internal error: ${(err && err.message) || 'unknown'}`);
		}
	}

	function sessionIdFor(req) {
		let id = req.headers['mcp-session-id'];
		if (typeof id !== 'string' || !id) id = null;
		if (!id) return null;
		if (!sessions.has(id)) {
			if (sessions.size >= MAX_SESSIONS) sessions.delete(sessions.keys().next().value);
			sessions.set(id, Date.now());
		}
		return id;
	}

	const handleRequest = async (req, res) => {
		let url;
		try {
			url = new URL(req.url || '/', 'http://localhost');
		} catch {
			send(res, 400, { error: 'invalid request URL' });
			return;
		}
		if (req.method === 'OPTIONS') {
			send(res, 204, undefined);
			return;
		}
		if (url.pathname === '/health') {
			// Unauthenticated: only what a launcher needs to recognise this server. The pid, the working
			// directory and the token stay behind the bearer token.
			const base = { ok: true, server: SERVER_NAME, version: SERVER_VERSION };
			if (isAuthorized(req.headers.authorization, token)) {
				send(res, 200, { ...base, pid: process.pid, started_at: startedAt, cwd, pi_version: piVersion });
			} else {
				send(res, 200, base);
			}
			return;
		}
		if (url.pathname !== '/mcp') {
			send(res, 404, { error: 'not found' });
			return;
		}
		if (!isAuthorized(req.headers.authorization, token)) {
			send(
				res,
				401,
				rpcError(null, -32001, 'unauthorized: send Authorization: Bearer <local_mcp token>'),
				{ 'WWW-Authenticate': 'Bearer realm="local_mcp"' },
			);
			return;
		}
		if (req.method === 'GET') {
			// No server-initiated stream: MCP allows 405 for a server that only answers POST.
			send(res, 405, { error: 'this server does not offer an SSE stream; POST JSON-RPC to /mcp' }, { Allow: 'POST, DELETE, OPTIONS' });
			return;
		}
		if (req.method === 'DELETE') {
			const id = sessionIdFor(req, res);
			if (id) sessions.delete(id);
			send(res, 204, undefined);
			return;
		}
		if (req.method !== 'POST') {
			send(res, 405, { error: 'method not allowed' }, { Allow: 'POST, DELETE, OPTIONS' });
			return;
		}

		let body;
		try {
			body = await readBody(req);
		} catch (err) {
			send(res, 413, rpcError(null, -32600, err.message));
			return;
		}
		let payload;
		try {
			payload = JSON.parse(body);
		} catch {
			send(res, 200, rpcError(null, -32700, 'Parse error: body is not valid JSON'));
			return;
		}

		let id = sessionIdFor(req);
		const controller = new AbortController();
		res.on('close', () => {
			// The client went away mid-call: abort the tool so a long bash command does not outlive it.
			if (!res.writableEnded) controller.abort();
		});
		const state = {
			controller,
			protocolVersion:
				typeof req.headers['mcp-protocol-version'] === 'string' ? req.headers['mcp-protocol-version'] : '',
		};

		const requestHeaders = () => {
			const headers = {};
			if (id) headers['Mcp-Session-Id'] = id;
			if (state.protocolVersion) headers['MCP-Protocol-Version'] = state.protocolVersion;
			return headers;
		};

		const isInitialize = !Array.isArray(payload) && !!payload && payload.method === 'initialize';
		if (isInitialize && !id) {
			// initialize opens the session; a client that already holds an id keeps it (also across restarts).
			id = `${SERVER_NAME}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
			if (sessions.size >= MAX_SESSIONS) sessions.delete(sessions.keys().next().value);
			sessions.set(id, Date.now());
		}

		if (Array.isArray(payload)) {
			if (payload.length === 0) {
				send(res, 200, rpcError(null, -32600, 'Invalid Request: empty batch'));
				return;
			}
			const responses = [];
			for (const message of payload) {
				const response = await dispatch(message, state);
				if (response) responses.push(response);
			}
			send(res, 200, responses.length ? responses : undefined, requestHeaders());
			return;
		}

		const response = await dispatch(payload, state);
		if (!response) {
			send(res, 202, undefined, requestHeaders());
			return;
		}
		send(res, 200, response, requestHeaders());
	};
	const server = createServer(handleRequest);
	const servers = [server];
	let stopping = false;
	let extraRetryTimer = null;

	server.on('error', (err) => {
		if (err.code === 'EADDRINUSE') {
			fatal(`${host}:${port} is already in use (another service owns it); refusing to take it over`);
		}
		fatal(`server error: ${err.message}`);
	});
	server.keepAliveTimeout = 65000;
	server.headersTimeout = 70000;
	server.listen(port, host, () => {
		logLine(`listening on http://${host}:${port}/mcp (cwd=${cwd}, pi=${piVersion}, pid=${process.pid}, ` +
			`tools=${[...tools.keys()].join(',')})`);
		if (!extraHost) return;
		let loggedMissing = false;
		const tryListenExtra = () => {
			if (stopping) return;
			const extra = createServer(handleRequest);
			extra.keepAliveTimeout = server.keepAliveTimeout;
			extra.headersTimeout = server.headersTimeout;
			extra.on('error', (err) => {
				if (servers.includes(extra)) servers.splice(servers.indexOf(extra), 1);
				if (stopping) return;
				const retryable = err.code === 'EADDRNOTAVAIL';
				if (!loggedMissing) {
					const why = retryable ? 'address not available yet (will retry every 15s until Tailscale is up)' : err.code || err.message;
					logLine(`extra listener ${extraHost}:${port} not started: ${why}; still serving ${host}:${port}`);
					loggedMissing = true;
				}
				try { extra.close(); } catch {}
				if (retryable) {
					extraRetryTimer = setTimeout(tryListenExtra, 15000);
					extraRetryTimer.unref();
				}
			});
			extra.listen(port, extraHost, () => {
				servers.push(extra);
				logLine(`also listening on http://${extraHost}:${port}/mcp`);
			});
		};
		tryListenExtra();
	});

	const shutdown = (signal) => {
		stopping = true;
		if (extraRetryTimer) clearTimeout(extraRetryTimer);
		logLine(`stopping on ${signal}`);
		let open = servers.length;
		for (const each of servers) each.close(() => --open === 0 && process.exit(0));
		// A hung connection must not keep the process alive; the launcher waits for the port to close.
		setTimeout(() => process.exit(0), 2000).unref();
	};
	process.on('SIGTERM', () => shutdown('SIGTERM'));
	process.on('SIGINT', () => shutdown('SIGINT'));
	process.on('uncaughtException', (err) => {
		logLine(`uncaught exception: ${err && err.stack ? err.stack : err}`);
		process.exit(1);
	});
	process.on('unhandledRejection', (err) => {
		logLine(`unhandled rejection: ${err && err.stack ? err.stack : err}`);
		process.exit(1);
	});
}

main();