"""Self-signed TLS certificate generation, no OpenSSL binary required.

Why this exists: mobile browsers (Chrome on Android, Safari on iOS) only
grant camera access (`getUserMedia`) on a "secure context." A plain LAN URL
like `http://192.168.1.23:8420` does NOT qualify, so the dashboard's camera
capture would silently fail on a phone. A self-signed HTTPS certificate
covering the server's actual LAN IP(s) makes the browser treat the
connection as secure once the user accepts the one-time "not private"
warning — the standard workaround for LAN device testing without a real CA.
"""
from __future__ import annotations

import datetime
import ipaddress
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from utils.logger import get_logger

log = get_logger("certs")


def get_lan_ip() -> str:
    """Best-effort discovery of this machine's LAN-facing IP (no packets sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_self_signed_cert(cert_path, key_path, lan_ip: str) -> None:
    if cert_path.exists() and key_path.exists():
        try:
            existing = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = existing.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            covered_ips = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
            if lan_ip in covered_ips and existing.not_valid_after_utc > datetime.datetime.now(datetime.timezone.utc):
                return  # existing cert already covers the current LAN IP and isn't expired
        except Exception:
            log.warning("Existing cert unreadable/stale, regenerating")

    log.info("Generating self-signed TLS certificate for LAN IP %s", lan_ip)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "omni-mind-atm-security.local")])
    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except ValueError:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    log.info("Certificate written to %s", cert_path)
