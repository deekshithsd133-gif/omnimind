"""JWT auth dependencies + role-based access control."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from database.db import get_session
from database.models import User
from utils.security import create_access_token, decode_access_token, verify_password

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def authenticate_user(username: str, password: str) -> User | None:
    with get_session() as db:
        user = db.query(User).filter(User.username == username).first()
        if user is None or not verify_password(password, user.password_hash):
            return None
        db.expunge(user)
        return user


def issue_token(user: User) -> str:
    return create_access_token(subject=user.username, role=user.role)


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return {"username": payload["sub"], "role": payload["role"]}


def require_role(*roles: str):
    def _dependency(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient permissions")
        return user

    return _dependency
