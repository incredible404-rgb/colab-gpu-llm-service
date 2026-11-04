# 用完点击按钮释放 GPU；固定 Tailscale Service、Magpie 配置和云盘文件保留。
# tailscale down 只让当前临时节点下线，不是 logout，不删除固定 Service。
# 释放后 Python 输出不再更新；以“管理会话”中没有活动会话为完成依据。
import ipywidgets as widgets
import os
import subprocess
from google.colab import runtime
from IPython.display import display
from html import escape

stop_button = widgets.Button(description='结束 GPU 运行时', button_style='warning')
stop_status = widgets.HTML(value='点击后关闭模型并释放 GPU；固定服务地址与配置保留。')

def stop_runtime(_):
    stop_button.disabled = True
    stop_button.description = '已请求释放'
    stop_status.value = ('正在关闭服务并释放 GPU。此提示断开后不再刷新；'
                         '请在“代码执行程序 → 管理会话”确认没有活动会话。')
    for name in ('qwen_server', 'tunnel_proc'):
        process = globals().get(name)
        if process is not None and process.poll() is None:
            process.terminate()
    tsctl = '/content/tsbin/tailscale'
    if os.path.exists(tsctl):
        try:
            subprocess.run([tsctl, '--socket=/var/run/tailscale/tailscaled.sock', 'down'],
                           timeout=5, capture_output=True)
        except Exception as error:
            print('节点下线未确认，将直接释放运行时：', type(error).__name__)
    try:
        runtime.unassign()
    except Exception as error:
        stop_button.disabled = False
        stop_button.description = '重试释放 GPU'
        stop_status.value = ('<b>释放请求失败：</b>' + escape(str(error)) +
                             '<br>请重试，或使用“代码执行程序 → 断开连接并删除运行时”。')

stop_button.on_click(stop_runtime)
display(widgets.VBox([stop_button, stop_status]))
