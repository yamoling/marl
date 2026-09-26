"""Workspace metadata and selection."""

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
