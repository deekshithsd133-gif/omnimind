from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_login_rejects_bad_credentials():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_login_succeeds_and_returns_role():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "admin"
    assert body["access_token"]


def test_protected_endpoint_requires_token():
    resp = client.get("/api/health")
    assert resp.status_code == 401


def test_protected_endpoint_accepts_valid_token():
    login = client.post("/api/auth/login", json={"username": "operator", "password": "operator123"})
    token = login.json()["access_token"]
    resp = client.get("/api/health", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_enroll_requires_admin_role():
    login = client.post("/api/auth/login", json={"username": "operator", "password": "operator123"})
    token = login.json()["access_token"]
    resp = client.post(
        "/api/enroll",
        headers={"Authorization": f"Bearer {token}"},
        data={"label": "test"},
        files={"file": ("x.jpg", b"not a real image", "image/jpeg")},
    )
    assert resp.status_code == 403
