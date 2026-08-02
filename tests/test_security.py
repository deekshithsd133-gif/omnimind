from utils.security import (
    create_access_token,
    decode_access_token,
    decrypt_bytes,
    encrypt_bytes,
    hash_password,
    verify_password,
)


def test_password_hash_roundtrip():
    h = hash_password("correct-horse-battery-staple")
    assert verify_password("correct-horse-battery-staple", h) is True
    assert verify_password("wrong-password", h) is False


def test_jwt_roundtrip_carries_role():
    token = create_access_token(subject="admin", role="admin")
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "admin"
    assert payload["role"] == "admin"


def test_jwt_rejects_garbage_token():
    assert decode_access_token("not.a.valid.token") is None


def test_field_encryption_roundtrip():
    original = b"\x00\x01\x02 some embedding bytes \xff\xfe"
    encrypted = encrypt_bytes(original)
    assert encrypted != original
    assert decrypt_bytes(encrypted) == original
