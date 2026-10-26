"""Checks of the account forms before anything is sent to the server."""

import re

MIN_PASSWORD_LENGTH = 6
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    """Loose format check: something@domain.zone without spaces."""
    return bool(EMAIL_PATTERN.match(email.strip()))


def is_valid_password(password: str) -> bool:
    """The server accepts passwords of at least MIN_PASSWORD_LENGTH characters."""
    return len(password) >= MIN_PASSWORD_LENGTH


def registration_error(
    first_name: str, last_name: str, email: str, password: str, repeat: str
) -> str | None:
    """Return the first problem of the registration form, or None if it is valid."""
    if not first_name.strip() or not last_name.strip():
        return "Укажите имя и фамилию."
    if not is_valid_email(email):
        return "Укажите корректный адрес электронной почты."
    if not is_valid_password(password):
        return f"Пароль должен содержать не менее {MIN_PASSWORD_LENGTH} символов."
    if password != repeat:
        return "Пароли не совпадают."
    return None


def new_password_error(new_password: str, repeat: str) -> str | None:
    """Return the problem of a new password, or None if it can be saved."""
    if not is_valid_password(new_password):
        return f"Новый пароль должен содержать не менее {MIN_PASSWORD_LENGTH} символов."
    if new_password != repeat:
        return "Пароли не совпадают."
    return None
