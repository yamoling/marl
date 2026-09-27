"""Workspace metadata and selection."""

import os
from pathlib import Path

from fastapi import APIRouter, Request

from ..errors import bad_request, json_response
from . import JsonBody

router = APIRouter(prefix="/api/workspaces")


def _body(body, field):
    """Require exactly one input field for a workspace mutation. @ai-generated"""
    if not isinstance(body, dict) or set(body) != {field}:
        raise bad_request(f"Expected {{{field}}}")
    return body[field]


@router.get("")
def list_workspaces(request: Request):
    """@ai-generated"""
    return json_response(request.app.state.workspaces.listing())


@router.get("/directories")
def browse_directories(request: Request, path: str | None = None):
    """List server-side directories for the local workspace root picker. @ai-generated"""
    workspaces = request.app.state.workspaces
    with workspaces.lock:
        initial = workspaces.items[workspaces.selected]["logdir"] if workspaces.selected else str(workspaces.default_root)
    target = Path(path if path is not None else initial).expanduser()
    if not target.is_absolute() or not target.is_dir():
        raise bad_request("path must be an existing absolute directory")
    target = target.resolve()
    try:
        with os.scandir(target) as entries:
            directories = sorted(
                ({"name": entry.name, "path": str(Path(entry.path).resolve())} for entry in entries if entry.is_dir()),
                key=lambda entry: entry["name"].casefold(),
            )
    except OSError as exc:
        raise bad_request(f"Cannot browse directory: {exc.strerror}") from exc
    parent = target.parent if target.parent != target else None
    return json_response({"path": str(target), "parent": str(parent) if parent else None, "directories": directories})


@router.post("", status_code=201)
def create_workspace(request: Request, body: JsonBody = None):
    """Create a workspace with an optional root logdir. @ai-edited"""
    if not isinstance(body, dict) or set(body) not in ({"name"}, {"name", "logdir"}):
        raise bad_request("Expected {name, logdir?}")
    return json_response(request.app.state.workspaces.create(body["name"], body.get("logdir")), 201)


@router.patch("/{workspace_id}")
def rename_workspace(workspace_id: str, request: Request, body: JsonBody = None):
    """@ai-generated"""
    return json_response(request.app.state.workspaces.rename(workspace_id, _body(body, "name")))


@router.delete("/{workspace_id}")
def delete_workspace(workspace_id: str, request: Request):
    """Trash workspace settings only; never delete experiment files. @ai-edited"""
    return json_response(request.app.state.workspaces.delete(workspace_id))


@router.post("/{workspace_id}/select")
def select_workspace(workspace_id: str, request: Request):
    """Select a workspace and its logs root. @ai-edited"""
    return json_response(request.app.state.workspaces.select(workspace_id))


@router.patch("/{workspace_id}/logdir")
def set_workspace_logdir(workspace_id: str, request: Request, body: JsonBody = None):
    """Update a workspace's root logdir. @ai-generated"""
    return json_response(request.app.state.workspaces.set_logdir(workspace_id, _body(body, "logdir")))
