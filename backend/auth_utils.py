from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from beanie import PydanticObjectId
from bson.errors import InvalidId
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings
from models.user import Role, User

JWT_ALGORITHM = "HS256"
MAX_PASSWORD_BYTES = 72  # bcrypt limit

_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt(rounds=get_settings().bcrypt_rounds)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, password_hash: str) -> bool:
    encoded = password.encode()
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False
    return bcrypt.checkpw(encoded, password_hash.encode())


def create_access_token(user: User) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "role": user.role.value,
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expiry_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


async def user_from_token(token: str) -> User | None:
    """Resolve a JWT to its user. Shared by HTTP routes and (later) socket auth."""
    try:
        payload = jwt.decode(token, get_settings().jwt_secret, algorithms=[JWT_ALGORITHM])
        user_id = PydanticObjectId(payload["sub"])
    except (jwt.PyJWTError, KeyError, InvalidId, TypeError):
        return None
    return await User.get(user_id)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    user = await user_from_token(credentials.credentials) if credentials else None
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*roles: Role) -> Callable[[User], Awaitable[User]]:
    """Dependency factory. The role is read from the database, not from the token."""

    async def dependency(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not permitted")
        return user

    return dependency
