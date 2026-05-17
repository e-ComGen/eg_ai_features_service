from fastapi import Header, HTTPException, status
from .config import INTERNAL_SERVICE_SECRET

async def verify_internal_token(x_internal_secret: str = Header(...)):
    """
    Проверяет, что пришел верный секретный ключ от Оркестратора.
    Если нет — пошел вон (403 Error).
    """
    if x_internal_secret != INTERNAL_SERVICE_SECRET:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Wrong secret key"
        )