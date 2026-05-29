"""Detailed proxy debug — raw socket, full HTTP CONNECT visibility."""
import os, sys, socket, base64
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

USER = "087cb323dfc815e7dec4"
PASS = "57c2f7d0e24d5631"
HOST = "go.proxycove.com"
PORT = 824

# Make HTTP CONNECT request manually and see proxy's response
print(f"Connecting to {HOST}:{PORT}...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(15)
try:
    s.connect((HOST, PORT))
    print("TCP connected.")
    auth = base64.b64encode(f"{USER}:{PASS}".encode()).decode()
    # Try HTTPS CONNECT first
    req = (
        f"CONNECT ifconfig.me:443 HTTP/1.1\r\n"
        f"Host: ifconfig.me:443\r\n"
        f"Proxy-Authorization: Basic {auth}\r\n"
        f"User-Agent: test/1.0\r\n"
        f"\r\n"
    )
    print(f"Sending CONNECT request:\n{req}")
    s.sendall(req.encode())
    # Read response
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
        if len(buf) > 8000:
            break
    print(f"Proxy responded ({len(buf)} bytes):")
    print(buf.decode('utf-8', errors='replace')[:1500])
except socket.timeout:
    print("TIMEOUT before proxy responded (proxy not answering)")
except Exception as e:
    print(f"ERR: {type(e).__name__}: {e}")
finally:
    s.close()
