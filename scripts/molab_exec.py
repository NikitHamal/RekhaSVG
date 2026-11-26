#!/usr/bin/env python3
"""
Clean MoLab REST client for https://sb-512e1e9b6c4d922b.sb.molab.run
Drives /api/kernel/execute with Marimo-Session-Id + Bearer token from /tmp/molab_token.
Never logs or stores tokens in tracked files.
"""
import base64
import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

MOLAB_URL = os.environ.get("MOLAB_URL", "https://sb-512e1e9b6c4d922b.sb.molab.run").rstrip("/")
TOKEN_FILE = "/tmp/molab_token"
SESSION_FILE = "/tmp/molab_session_id"


def get_token() -> str:
    if os.environ.get("MOLAB_TOKEN"):
        return os.environ["MOLAB_TOKEN"].strip()
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise RuntimeError("Missing /tmp/molab_token")


def get_session_id() -> str:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            s = f.read().strip()
            if s:
                return s
    s = "s_rekhasvg01"
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        f.write(s)
    return s


def open_ws_session(session_id: str):
    token = get_token()
    parsed = urllib.parse.urlparse(MOLAB_URL)
    host = parsed.hostname
    port = parsed.port or 443
    ctx = ssl.create_default_context()
    raw_sock = socket.create_connection((host, port), timeout=15)
    tls_sock = ctx.wrap_socket(raw_sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET /ws?session_id={session_id}&file=/marimo/notebook.py HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Authorization: Bearer {token}\r\n"
        f"User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n\r\n"
    )
    tls_sock.sendall(req.encode("utf-8"))
    tls_sock.recv(4096)
    return tls_sock


def http_req(path: str, method: str = "GET", payload: dict = None, timeout: int = 120):
    token = get_token()
    session_id = get_session_id()
    url = f"{MOLAB_URL}{path}"
    req_headers = {
        "Authorization": f"Bearer {token}",
        "Marimo-Session-Id": session_id,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream, text/plain, */*",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, dict(e.headers), body


def parse_sse(raw: str) -> dict:
    stdout_parts = []
    stderr_parts = []
    done_data = {}
    current_event = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            d_str = line.split(":", 1)[1].strip()
            try:
                obj = json.loads(d_str)
            except Exception:
                obj = {"data": d_str}
            if current_event == "stdout":
                stdout_parts.append(obj.get("data", ""))
            elif current_event == "stderr":
                stderr_parts.append(obj.get("data", ""))
            elif current_event == "done":
                done_data = obj
    return {
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "done": done_data,
        "raw": raw,
    }


def execute_python(code: str, timeout: int = 180) -> dict:
    session_id = get_session_id()
    ws_sock = None
    try:
        ws_sock = open_ws_session(session_id)
        st, _, body = http_req("/api/kernel/execute", method="POST", payload={"code": code}, timeout=timeout)
        if st != 200:
            return {"status": st, "stdout": "", "stderr": body, "done": {"success": False}}
        res = parse_sse(body)
        res["status"] = st
        return res
    finally:
        if ws_sock is not None:
            try:
                ws_sock.close()
            except Exception:
                pass


def execute_bash(cmd: str, timeout: int = 180) -> dict:
    b64_cmd = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    py_code = f"""
import base64, subprocess, sys
cmd = base64.b64decode("{b64_cmd}").decode("utf-8")
p = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True)
if p.stdout:
    sys.stdout.write(p.stdout)
if p.stderr:
    sys.stderr.write(p.stderr)
if p.returncode != 0:
    print(f"[EXIT_CODE={{p.returncode}}]", file=sys.stderr)
"""
    return execute_python(py_code, timeout=timeout)


def upload_file(local_path: str, remote_path: str, mode: int = 0o644) -> dict:
    with open(local_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("ascii")
    py_code = f"""
import base64, os
os.makedirs(os.path.dirname("{remote_path}") or "/", exist_ok=True)
with open("{remote_path}", "wb") as f:
    f.write(base64.b64decode("{b64_data}"))
os.chmod("{remote_path}", {mode})
print("UPLOADED:", "{remote_path}", os.path.getsize("{remote_path}"))
"""
    return execute_python(py_code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: molab_exec.py [py|sh|upload|status] ...")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "status":
        st, _, body = http_req("/api/status")
        print(st, body)
    elif mode == "py":
        res = execute_python(sys.argv[2])
        if res["stdout"]:
            sys.stdout.write(res["stdout"])
        if res["stderr"]:
            sys.stderr.write(res["stderr"])
    elif mode == "sh":
        res = execute_bash(sys.argv[2])
        if res["stdout"]:
            sys.stdout.write(res["stdout"])
        if res["stderr"]:
            sys.stderr.write(res["stderr"])
    elif mode == "upload":
        m_oct = int(sys.argv[4], 8) if len(sys.argv) > 4 else 0o644
        res = upload_file(sys.argv[2], sys.argv[3], m_oct)
        print(res["stdout"], res["stderr"])
