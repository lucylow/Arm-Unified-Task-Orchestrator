from fastapi import Request, HTTPException
from .store import users_db


def get_user(request: Request):
    username = request.headers.get("x-username")
    if not username or username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid user")
    # return the username and the record for clarity
    return username, users_db[username]
