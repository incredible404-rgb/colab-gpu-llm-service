"""Real launcher/Node bind regression: local health and the cloud-facing interface must agree.

Uses only temporary config, random ports and two loopback addresses; never allocates a GPU.
Run with the same Python used by the launcher. Node and an installed Pi are required.
"""
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from unittest import mock

launcher = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('local_mcp.py')
spec = importlib.util.spec_from_file_location('bind_check_launcher', launcher)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def healthy(host, port):
    connection = http.client.HTTPConnection(host, port, timeout=1)
    try:
        connection.request('GET', '/health')
        response = connection.getresponse()
        body = json.loads(response.read())
        return response.status == 200 and body.get('server') == 'local_mcp'
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def check(name, configured_ip, explicit_host, expected_second, expected_local=True):
    with tempfile.TemporaryDirectory(prefix='local-mcp-bind-') as temp:
        root = Path(temp)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        env = {key: value for key, value in os.environ.items()
               if key not in ('LOCAL_MCP_HOST', 'LOCAL_MCP_EXTRA_HOST', 'LOCAL_TAILSCALE_IP')}
        env.update(LOCAL_MCP_CONFIG=str(root/'config.json'), LOCAL_MCP_PORT=str(port),
                   LOCAL_MCP_LOG=str(root/'server.log'), LOCAL_MCP_CWD=str(root))
        if explicit_host:
            env['LOCAL_MCP_HOST'] = explicit_host
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(helper, 'tailscale_bind_ip', return_value=configured_ip):
            try:
                result = helper.start()
                assert result['ok'], 'Isolated launcher failed to start'
                reused = helper.start()
                assert reused['ok'] and not reused['started'] and reused['pid'] == result['pid'], 'Start must reuse the owned process'
                assert helper.status()['running'], 'Status must probe the bound interface'
                local = healthy('127.0.0.1', port)
                second = healthy('127.0.0.2', port)
                print(json.dumps({'case':name, 'local_health':local, 'second_interface_health':second}), flush=True)
                assert local is expected_local, 'Local listening must match the requested binding'
                assert second is expected_second, 'Configured cloud-facing interface does not match the listener'
            finally:
                stopped = helper.stop()
                assert stopped['ok'], 'Failed to stop the isolated owned bridge'


check('configured interface', '127.0.0.2', '', True)
check('no configured interface', '', '', False)
check('explicit loopback overrides configured interface', '127.0.0.2', '127.0.0.1', False)
check('explicit interface overrides loopback', '127.0.0.1', '127.0.0.2', True, expected_local=False)
print(json.dumps({'ok':True,'gpu_allocated':False,'user_config_modified':False}))
