"""Episode replay through `marl` (full deserialization on purpose)."""

from http import HTTPStatus

import orjson

from ..data.library import Library
from ..data.records import ExperimentRecord, RunRecord
from ..errors import ApiError, conflict, not_found


def ensure_replayable(library: Library, record: ExperimentRecord) -> ExperimentRecord:
    """Run the lazy health check if needed; 409 with the blocking issue unless `replay` is true. @ai-generated"""
    if record.capabilities.replay is None:
        library.health(record.id)
    record = library.get(record.id) or record
    if record.capabilities.replay is not True:
        blocking = next((i for i in record.issues if i.code in ("deserialize-failed", "missing-keys", "missing-experiment-json")), None)
        raise conflict("The experiment cannot be replayed with the current marl code", "not-replayable", blocking)
    return record


def replay(library: Library, record: ExperimentRecord, run: RunRecord, step: int, test: int, only_saved_actions: bool) -> bytes:
    """
    Replay a test episode and serialize it exactly like the old `/experiment/replay/...` route:
    `Run.replay_episode` + orjson with `marl.utils.default_serialization`. Blocking: call it from a
    worker thread.

    @ai-generated
    """
    ensure_replayable(library, record)
    import marl
    from marl.models.run import Run

    try:
        full_run = Run.load(run.path)
        # The actual location is authoritative, not the serialized `rundir`.
        full_run.rundir = str(run.path)
        episode = full_run.replay_episode(step, test, only_saved_actions=only_saved_actions)
    except FileNotFoundError as exc:
        raise not_found(str(exc), "replay-unavailable") from exc
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "replay-failed", f"{type(exc).__name__}: {exc}") from exc
    return orjson.dumps(episode, option=orjson.OPT_SERIALIZE_NUMPY, default=marl.utils.default_serialization)
