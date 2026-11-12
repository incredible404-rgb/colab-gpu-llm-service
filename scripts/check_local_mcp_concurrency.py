"""Regression: the manual button and keeper may launch the same bridge concurrently."""
import concurrent.futures
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile

launcher = Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix='local-mcp-concurrent-') as tmp:
    root = Path(tmp)
    config = root / 'config.json'
    config.write_text(json.dumps({'token': secrets.token_urlsafe(32), 'cwd': str(root)}), encoding='utf-8')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, LOCAL_MCP_CONFIG=str(config), LOCAL_MCP_PORT=str(port), LOCAL_MCP_LOG=str(root / 'server.log'))

    def run(command):
        completed = subprocess.run([sys.executable, '-X', 'utf8', str(launcher), command], env=env,
                                   capture_output=True, text=True, encoding='utf-8', timeout=20)
        return json.loads(completed.stdout.strip().splitlines()[-1])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(run, ['start'] * 3))
        print(json.dumps(results, ensure_ascii=False))
        assert all(item.get('ok') for item in results), 'Concurrent starts must all reuse the healthy service'
        assert len({item.get('pid') for item in results}) == 1, 'Concurrent starts must converge on one process'
        print('PASS: three concurrent launchers converged on one healthy MCP')
    finally:
        stopped = run('stop')
        assert stopped.get('ok'), stopped
