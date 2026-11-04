"""并行分块下载 Drive 文件。文件上半部分会被原样注入 Colab 笔记本单元格。"""
import concurrent.futures
import time
from pathlib import Path

import requests

CHUNK_BYTES = 64 * 1024 * 1024
THREADS = 8


def drive_media_url(file_id):
    return f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"


def bearer_headers(credentials):
    if not credentials.valid:
        from google.auth.transport.requests import Request

        credentials.refresh(Request())
    return {"Authorization": f"Bearer {credentials.token}"}


def download_range(url, headers, start, end, path, attempts=5):
    expected = end - start + 1
    last = None
    for attempt in range(attempts):
        try:
            ranged = dict(headers, **{"Range": f"bytes={start}-{end}"})
            with requests.get(url, headers=ranged, stream=True, timeout=180) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"服务端未按范围响应：HTTP {response.status_code}")
                written = 0
                with open(path, "r+b") as handle:
                    handle.seek(start)
                    for block in response.iter_content(4 * 1024 * 1024):
                        handle.write(block)
                        written += len(block)
            if written != expected:
                raise RuntimeError(f"分块 {start}-{end} 写入 {written} 字节，预期 {expected}")
            return
        except Exception as error:  # 重试若干次后向外抛出
            last = error
            time.sleep(2**attempt)
    raise RuntimeError(f"分块下载失败 {start}-{end}: {last!r}")


def download_parallel(url, headers, size, path, threads=THREADS, chunk=CHUNK_BYTES, log=print, quiet_seconds=20, attempts=5):
    if threads < 1:
        raise ValueError("threads 必须 >= 1")
    if size <= 0:
        raise ValueError("size 必须为正")
    with open(path, "wb") as handle:
        handle.truncate(size)
    ranges = [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]
    finished = 0
    report = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        pending = [pool.submit(download_range, url, headers, start, end, str(path), attempts) for start, end in ranges]
        for future in concurrent.futures.as_completed(pending):
            future.result()
            finished += 1
            if log and time.monotonic() - report >= quiet_seconds:
                log(f"已读取 {finished / len(ranges):.0%}")
                report = time.monotonic()
    actual = path.stat().st_size
    if actual != size:
        raise RuntimeError(f"{path.name} 大小不符：{actual} != {size}")
    return path


def download_many_parallel(
    jobs,
    threads=THREADS,
    chunk=CHUNK_BYTES,
    log=print,
    quiet_seconds=10,
    attempts=5,
):
    """用同一个线程池并发读取多个文件。

    jobs 中每一项至少包含 url、headers、size 和 path。大文件会拆成多个
    范围请求；多个文件共享同一并发上限，避免按文件数成倍放大连接数。
    """
    if threads < 1:
        raise ValueError("threads 必须 >= 1")
    if not jobs:
        return []

    prepared = []
    tasks = []
    total_bytes = 0
    for index, job in enumerate(jobs):
        size = int(job["size"])
        if size <= 0:
            raise ValueError(f"{job['path']} 大小必须为正")
        target = Path(job["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.truncate(size)
        file_ranges = [(start, min(start + chunk, size) - 1) for start in range(0, size, chunk)]
        prepared.append((job, target, size, file_ranges))
        total_bytes += size
        for start, end in file_ranges:
            tasks.append((index, job["url"], job["headers"], start, end, target, end - start + 1))

    finished_bytes = {index: 0 for index in range(len(prepared))}
    finished_ranges = {index: 0 for index in range(len(prepared))}
    completed_bytes = 0
    report = time.monotonic()
    started = report
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        pending = {
            pool.submit(download_range, url, headers, start, end, str(target), attempts): (index, byte_count)
            for index, url, headers, start, end, target, byte_count in tasks
        }
        for future in concurrent.futures.as_completed(pending):
            index, byte_count = pending[future]
            future.result()
            finished_bytes[index] += byte_count
            finished_ranges[index] += 1
            completed_bytes += byte_count
            job, target, size, file_ranges = prepared[index]
            if finished_ranges[index] == len(file_ranges):
                callback = job.get("on_file_done")
                if callback:
                    callback(target)
            if log and time.monotonic() - report >= quiet_seconds:
                elapsed = max(time.monotonic() - started, 0.001)
                rate = completed_bytes / elapsed / (1024 * 1024)
                log(f"已读取 {completed_bytes / total_bytes:.0%}，平均 {rate:.1f} MB/s")
                report = time.monotonic()

    elapsed = time.monotonic() - started
    results = []
    for job, target, size, file_ranges in prepared:
        actual = target.stat().st_size
        if actual != size:
            raise RuntimeError(f"{target.name} 大小不符：{actual} != {size}")
        index = len(results)
        if finished_ranges[index] != len(file_ranges) or finished_bytes[index] != size:
            raise RuntimeError(f"{target.name} 分块未完整写入")
        results.append(target)
    if log:
        rate = total_bytes / max(elapsed, 0.001) / (1024 * 1024)
        log(f"读取完成：{total_bytes / (1024 ** 3):.2f} GiB，用时 {elapsed:.1f} 秒，平均 {rate:.1f} MB/s")
    return results


# ----------------------------- 本地自测 -----------------------------
def _selftest():
    import hashlib
    import http.server
    import tempfile
    import threading
    from pathlib import Path

    size = 48 * 1024 * 1024
    payload = bytes(range(256)) * (size // 256)
    digest = hashlib.sha256(payload).hexdigest()

    class Handler(http.server.BaseHTTPRequestHandler):
        ignore_range = False
        truncate = 0
        seen = []
        live = 0
        peak = 0

        def log_message(self, *args):
            pass

        def do_GET(self):
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)
            try:
                raw = self.headers.get("Range")
                type(self).seen.append(raw)
                if raw and not type(self).ignore_range:
                    first, last = raw.replace("bytes=", "").split("-")
                    first, last = int(first), min(int(last), size - 1)
                    body = payload[first : last + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
                else:
                    body = payload
                    self.send_response(200)
                if type(self).truncate:
                    body = body[: type(self).truncate]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                type(self).live -= 1

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/blob"
    chunk = 6 * 1024 * 1024

    def run(threads, path):
        Handler.seen.clear()
        Handler.peak = 0
        started = time.monotonic()
        download_parallel(url, {}, size, path, threads=threads, chunk=chunk, log=None, quiet_seconds=0, attempts=1)
        return time.monotonic() - started

    try:
        with tempfile.TemporaryDirectory() as tmp:
            for threads in (1, 8):
                target = Path(tmp) / f"out-{threads}.bin"
                elapsed = run(threads, target)
                assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, f"threads={threads} 内容不一致"
                assert all(item for item in Handler.seen), "存在无 Range 的请求"
                print(f"threads={threads} 内容校验通过 ranges={len(Handler.seen)} peak={Handler.peak} {elapsed * 1000:.0f}ms")

            first = Path(tmp) / "many-a.bin"
            second = Path(tmp) / "many-b.bin"
            completed = []
            Handler.seen.clear()
            Handler.peak = 0
            started = time.monotonic()
            download_many_parallel(
                [
                    {"url": url, "headers": {}, "size": size, "path": first,
                     "on_file_done": lambda path: completed.append(path)},
                    {"url": url, "headers": {}, "size": size, "path": second,
                     "on_file_done": lambda path: completed.append(path)},
                ],
                threads=4,
                chunk=chunk,
                log=None,
                quiet_seconds=0,
                attempts=1,
            )
            assert hashlib.sha256(first.read_bytes()).hexdigest() == digest, "多文件第一项内容不一致"
            assert hashlib.sha256(second.read_bytes()).hexdigest() == digest, "多文件第二项内容不一致"
            assert Handler.peak > 1, "多文件读取没有形成并发"
            assert {Path(path).name for path in completed} == {"many-a.bin", "many-b.bin"}, completed
            print(f"多文件并发内容校验通过 ranges={len(Handler.seen)} peak={Handler.peak} {(time.monotonic() - started) * 1000:.0f}ms")

            Handler.ignore_range = True
            try:
                run(4, Path(tmp) / "ignored.bin")
            except RuntimeError as error:
                print("拒绝忽略 Range 的响应:", str(error)[:90])
            else:
                raise AssertionError("服务端忽略 Range 时应当报错")
            Handler.ignore_range = False

            Handler.truncate = 1024
            try:
                run(4, Path(tmp) / "short.bin")
            except RuntimeError as error:
                print("短块被拦下:", str(error)[:90])
            else:
                raise AssertionError("分块字节数不足时应当报错")
            Handler.truncate = 0
    finally:
        server.shutdown()
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
