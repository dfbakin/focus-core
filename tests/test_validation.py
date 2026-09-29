"""Tests for account form validation."""

import pytest

from app.core.validation import is_valid_email, new_password_error, registration_error


@pytest.mark.parametrize("email", ["a@b.ru", "teacher.name@school.edu"])
def test_valid_emails(email: str) -> None:
    assert is_valid_email(email)


@pytest.mark.parametrize("email", ["", "a@b", "a b@c.ru", "@c.ru"])
def test_invalid_emails(email: str) -> None:
    assert not is_valid_email(email)


def test_registration_checks_in_order() -> None:
    assert registration_error("", "И", "a@b.ru", "secret1", "secret1") is not None
    assert "почты" in registration_error("А", "И", "bad", "secret1", "secret1")
    assert "6" in registration_error("А", "И", "a@b.ru", "123", "123")
    assert "совпадают" in registration_error("А", "И", "a@b.ru", "secret1", "secret2")
    assert registration_error("А", "И", "a@b.ru", "secret1", "secret1") is None


def test_new_password() -> None:
    assert new_password_error("123", "123") is not None
    assert new_password_error("secret1", "other") == "Пароли не совпадают."
    assert new_password_error("secret1", "secret1") is None
