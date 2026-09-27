"""API errors and their `{error, message, issue?}` JSON body."""

from http import HTTPStatus
from pathlib import Path
from typing import Any

import orjson
from fastapi import Response

from .data.issues import Issue


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str, issue: Issue | None = None, detail: str | None = None):
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message
        self.issue = issue
        self.detail = detail

    def body(self) -> dict[str, Any]:
        """@ai-generated"""
        body: dict[str, Any] = {"error": self.error, "message": self.message}
        if self.issue is not None:
            body["issue"] = self.issue.to_json()
        if self.detail is not None:
            body["detail"] = self.detail
        return body


def bad_request(message: str, error: str = "invalid-request") -> ApiError:
    return ApiError(HTTPStatus.BAD_REQUEST, error, message)


def not_found(message: str, error: str = "not-found") -> ApiError:
    return ApiError(HTTPStatus.NOT_FOUND, error, message)


def forbidden(message: str, error: str = "forbidden") -> ApiError:
    return ApiError(HTTPStatus.FORBIDDEN, error, message)


def conflict(message: str, error: str = "conflict", issue: Issue | None = None) -> ApiError:
    return ApiError(HTTPStatus.CONFLICT, error, message, issue)


def _default(obj: Any) -> Any:
    """@ai-generated"""
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def json_response(content: Any, status: int = HTTPStatus.OK) -> Response:
    """orjson response; NaN/inf become null, numpy arrays are supported. @ai-generated"""
    body = orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY, default=_default)
    return Response(body, status_code=status, media_type="application/json")


def error_response(error: str, message: str, status: int, issue: Issue | None = None, detail: str | None = None) -> Response:
    """@ai-generated"""
    return json_response(ApiError(status, error, message, issue, detail).body(), status)
