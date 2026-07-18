#!/usr/bin/env python3
"""Evaluate JS in a CDP tab (awaitPromise) and print the string result."""
import json, sys, struct, socket, base64, os, hashlib

def ws_connect(host, port, path):
    s = socket.create_connection((host, port))
    key = base64.b64encode(os.urandom(16)).decode()
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
           f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    s.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(4096)
    assert b"101" in resp.split(b"\r\n")[0], resp[:200]
    return s

def ws_send(s, payload: bytes):
    header = bytearray([0x81])
    n = len(payload)
    mask = os.urandom(4)
    if n < 126: header.append(0x80 | n)
    elif n < 65536: header.append(0x80 | 126); header += struct.pack(">H", n)
    else: header.append(0x80 | 127); header += struct.pack(">Q", n)
    header += mask
    s.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

def _recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk: raise ConnectionError("closed")
        buf += chunk
    return buf

def ws_recv(s):
    while True:
        h = _recv_exact(s, 2)
        fin = h[0] & 0x80; opcode = h[0] & 0x0F
        n = h[1] & 0x7F
        if n == 126: n = struct.unpack(">H", _recv_exact(s, 2))[0]
        elif n == 127: n = struct.unpack(">Q", _recv_exact(s, 8))[0]
        data = _recv_exact(s, n) if n else b""
        if opcode == 0x9:  # ping -> pong
            ws_send(s, data); continue
        if opcode in (0x1, 0x2, 0x0):
            # assume no fragmentation for our sizes... accumulate if needed
            if not fin:
                frag = data
                while True:
                    h2 = _recv_exact(s, 2)
                    fin2 = h2[0] & 0x80; n2 = h2[1] & 0x7F
                    if n2 == 126: n2 = struct.unpack(">H", _recv_exact(s, 2))[0]
                    elif n2 == 127: n2 = struct.unpack(">Q", _recv_exact(s, 8))[0]
                    frag += _recv_exact(s, n2)
                    if fin2: break
                return frag
            return data

def evaluate(tab_id, expr, timeout=120):
    s = ws_connect("localhost", 9888, f"/devtools/page/{tab_id}")
    s.settimeout(timeout)
    msg = {"id": 1, "method": "Runtime.evaluate",
           "params": {"expression": expr, "awaitPromise": True,
                      "returnByValue": True, "timeout": timeout * 1000}}
    ws_send(s, json.dumps(msg).encode())
    while True:
        resp = json.loads(ws_recv(s))
        if resp.get("id") == 1:
            s.close()
            return resp

if __name__ == "__main__":
    tab = sys.argv[1]
    expr = open(sys.argv[2]).read() if len(sys.argv) > 2 else sys.stdin.read()
    r = evaluate(tab, expr)
    res = r.get("result", {}).get("result", {})
    if "exceptionDetails" in r.get("result", {}):
        print("EXCEPTION:", json.dumps(r["result"]["exceptionDetails"])[:2000], file=sys.stderr)
        sys.exit(1)
    v = res.get("value")
    print(v if isinstance(v, str) else json.dumps(v))
