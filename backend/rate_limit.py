from slowapi import Limiter
from slowapi.util import get_remote_address

from config import get_settings

limiter = Limiter(key_func=get_remote_address)


def auth_limit() -> str:
    return get_settings().rate_limit_auth
