"""Cell-1 tail shared by both Colab notebooks: ready banner, WebUI tab and stop button.

Browsers only let the frame that called window.open() close that tab, so the WebUI is opened
from cell 1's output frame and only cell 1's stop button closes it. The stop cell, the idle
auto-stop and Colab reclaiming the runtime leave the tab open: reaching it from there would
need the frame to keep polling the kernel, which is not worth it for one manual close.
"""


CONSOLE_TEMPLATE = r'''
# __STEP__. 就绪横幅 + WebUI 标签页 + 停机按钮
# 浏览器只允许 window.open 的发起帧关闭该标签页，所以 WebUI 由本格输出帧打开，本格停机按钮先关它再释放运行时。
from google.colab import output as colab_output, runtime
from IPython.display import HTML, display

def _qwen_stop():
    if globals().get('qwen_server') and qwen_server.poll() is None:
        qwen_server.terminate()
    if Path(TS).exists():
        try:
            subprocess.run([TS, '--socket=' + TS_SOCK, 'down'], timeout=5, capture_output=True)
        except Exception:
            pass
    runtime.unassign()

colab_output.register_callback('qwen.stop', _qwen_stop)

_QWEN_CONSOLE_HTML = """
<div style="margin:10px 0;padding:12px;border-radius:8px;background:#1e293b;color:#f8fafc;font-size:15px;line-height:1.6;">
  🚀 <b>__TITLE__</b>
  <a id="qwen-open" href="__URL__" style="color:#38bdf8;font-weight:bold;text-decoration:underline;margin-left:8px;">👉 点击打开 WebUI 交互界面 (免密直连)</a>
  <div style="margin-top:10px;">
    <button id="qwen-stop" style="background:#f0ad4e;color:#111;border:0;border-radius:4px;padding:6px 14px;font-size:14px;cursor:pointer;">🛑 结束 __GPU__ 运行时 (省 CU)</button>
  </div>
  <div id="qwen-state" style="margin-top:8px;font-size:14px;">__STATUS__</div>
</div>
<script>
(() => {
  const URL = "__URL__";
  const stateEl = document.getElementById("qwen-state");
  const bootMs = Date.now();
  let win = null;
  // 只由本次运行的 eval_js 调用；重开笔记本时旧输出里的脚本会重跑，但不会自动弹窗
  window.qwenOpenWebui = () => {
    win = window.open(URL, "qwen-webui");
    if (!win) stateEl.textContent = "浏览器拦截了自动弹窗，请点上面的链接打开";
  };
  document.getElementById("qwen-open").addEventListener("click", (e) => {
    e.preventDefault();
    window.qwenOpenWebui();
  });
  document.getElementById("qwen-stop").addEventListener("click", (e) => {
    e.currentTarget.disabled = true;
    e.currentTarget.textContent = "正在释放…";
    if (win && !win.closed) win.close();
    const runMin = ((Date.now() - bootMs) / 60000).toFixed(1);
    stateEl.innerHTML = "<b>正在关闭服务并销毁运行时（本次就绪后已运行 " + runMin + " 分钟，停止计费）…</b>";
    google.colab.kernel.invokeFunction("qwen.stop", [], {}).catch(() => {});
  });
})();
</script>
""".replace('__URL__', 'http://%s:443/' % SERVICE_FQDN)
display(HTML(_QWEN_CONSOLE_HTML))
try:
    colab_output.eval_js('window.qwenOpenWebui()', ignore_result=True)
except Exception:
    pass
'''


def console_code(step: int, title: str, gpu: str, status_html: str) -> str:
    """Return the cell-1 tail. Needs Path, subprocess, TS, TS_SOCK, SERVICE_FQDN in cell 1."""
    return (CONSOLE_TEMPLATE.replace('__STEP__', str(step))
            .replace('__TITLE__', title)
            .replace('__GPU__', gpu)
            .replace('__STATUS__', status_html))
