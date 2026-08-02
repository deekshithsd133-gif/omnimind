"""JWT auth, password hashing, and field-level encryption.

Note on "database encryption": SQLCipher has no reliable prebuilt wheel on
Windows without a local C build toolchain, so full-disk DB encryption isn't
realistic to guarantee in this environment. Instead we encrypt every
sensitive column (face embeddings, evidence file paths) at the application
layer with Fernet (AES-128-CBC + HMAC) before it ever reaches SQLite. Plain
face images are never persisted at all — only embeddings, which are
themselves encrypted at rest.
"""
from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from jose import JWTError, jwt
from passlib.context import CryptContext

from config.settings import settings
from utils.logger import get_logger

log = get_logger("security")

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _load_or_create_fernet_key() -> bytes:
    if settings.field_encryption_key:
        return settings.field_encryption_key.encode()

    key_path = settings.db_path.parent / "field_encryption.key"
    if key_path.exists():
        return key_path.read_bytes()

    key = Fernet.generate_key()
    key_path.write_bytes(key)
    log.warning("Generated new field-encryption key at %s — back this up; losing it makes stored embeddings unrecoverable.", key_path)
    return key


_fernet = Fernet(_load_or_create_fernet_key())


def encrypt_bytes(data: bytes) -> bytes:
    return _fernet.encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    return _fernet.decrypt(token)


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd_context.verify(password, password_hash)
    except Exception:
        log.exception("Password verification raised unexpectedly")
        return False


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None
