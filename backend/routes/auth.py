from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, StringConstraints, field_validator
from pymongo.errors import DuplicateKeyError

from auth_utils import (
    MAX_PASSWORD_BYTES,
    CurrentUser,
    create_access_token,
    hash_password,
    verify_password,
)
from models.user import Role, User
from rate_limit import auth_limit, limiter

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Indian mobile number: 10 digits, starting 6–9
IndianPhone = Annotated[str, StringConstraints(pattern=r"^[6-9]\d{9}$")]


@lru_cache
def _dummy_hash() -> str:
    """Compared against when the email is unknown, so timing doesn't reveal which emails exist."""
    return hash_password("no-such-user")


class RegisterRequest(BaseModel):
    """Patient registration. There is no role field: any `role` sent by a client is ignored."""

    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    phone: IndianPhone
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def _fits_bcrypt(cls, value: str) -> str:
        if len(value.encode()) > MAX_PASSWORD_BYTES:
            raise ValueError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    role: Role
    department: str | None = None

    @classmethod
    def from_user(cls, user: User) -> "UserOut":
        return cls(
            id=str(user.id),
            name=user.name,
            email=user.email,
            phone=user.phone,
            role=user.role,
            department=user.department,
        )


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105
    user: UserOut


@router.post("/register", status_code=status.HTTP_201_CREATED)
@limiter.limit(auth_limit)
async def register(request: Request, body: RegisterRequest) -> AuthResponse:
    user = User(
        name=body.name.strip(),
        email=body.email.lower(),
        phone=body.phone,
        password_hash=hash_password(body.password),
        role=Role.PATIENT,
    )
    try:
        await user.insert()
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        ) from None
    return AuthResponse(access_token=create_access_token(user), user=UserOut.from_user(user))


@router.post("/login")
@limiter.limit(auth_limit)
async def login(request: Request, body: LoginRequest) -> AuthResponse:
    user = await User.find_one(User.email == body.email.lower())
    password_ok = verify_password(body.password, user.password_hash if user else _dummy_hash())
    if user is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )
    return AuthResponse(access_token=create_access_token(user), user=UserOut.from_user(user))


@router.get("/me")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.from_user(user)
