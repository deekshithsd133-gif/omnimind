"""Entry point: ensures a LAN-valid self-signed TLS cert exists, then starts
uvicorn over HTTPS so phone browsers on the same WiFi can grant camera
access (see utils/certs.py for why plain HTTP won't work on mobile).
"""
from __future__ import annotations

import uvicorn

from config.settings import settings
from utils.certs import ensure_self_signed_cert, get_lan_ip
from utils.logger import get_logger

log = get_logger("run_server")

if __name__ == "__main__":
    cert_dir = settings.db_path.parent
    cert_path = cert_dir / "server.crt"
    key_path = cert_dir / "server.key"

    lan_ip = get_lan_ip()
    ensure_self_signed_cert(cert_path, key_path, lan_ip)

    url = f"https://{lan_ip}:{settings.port}"
    print("=" * 70)
    print(f"  Omni Mind ATM Security System")
    print(f"  Local:  https://127.0.0.1:{settings.port}")
    print(f"  Mobile (same WiFi): {url}")
    print(f"  NOTE: your browser will warn 'connection is not private' the")
    print(f"        first time — this is expected for a self-signed LAN")
    print(f"        certificate. Tap Advanced -> Proceed to continue.")
    print(f"  Default logins: admin/admin123, operator/operator123")
    print("=" * 70)

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        ssl_keyfile=str(key_path),
        ssl_certfile=str(cert_path),
        log_level="info",
    )
