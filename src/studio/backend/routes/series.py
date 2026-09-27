"""`POST /api/series`: batch of independent series queries with partial success."""

from fastapi import APIRouter

from .. import settings
from ..errors import bad_request, json_response
from . import JsonBody, LibraryDep

router = APIRouter(prefix="/api")


@router.post("/series")
def series(library: LibraryDep, body: JsonBody = None):
    """`{queries: SeriesQuery[]}` (1..200) -> `[{ok: true, result} | {ok: false, issue}]`. @ai-generated"""
    queries = body.get("queries") if isinstance(body, dict) else None
    if not isinstance(queries, list) or not 1 <= len(queries) <= settings.MAX_SERIES_QUERIES:
        raise bad_request(f"Expected {{queries: [...]}} with 1 to {settings.MAX_SERIES_QUERIES} queries")
    return json_response(library.series(queries))
