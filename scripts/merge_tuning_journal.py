"""
Merge Optuna journal files produced independently by two workstations tuning in parallel.

The journal is append-only and both machines write to their own copy of the same file, so `git`
ends up with a conflict once both sides are pushed. This script never re-runs any trial: it reads
each side with Optuna's own storage backend and re-emits every study/trial it finds into a fresh
journal via Optuna's `template_trial` mechanism, which preserves params, state, values, timestamps
and attributes exactly.

Studies are expected to have distinct names across machines (see `scripts/tuning.py`, which derives
the name from the algorithm and layout). If the same study name shows up on both sides, its trials
are *not* merged together: the local (this machine's) copy is kept as a separate study, renamed by
appending `-<hostname>` (e.g. "DQN-interdependent-2-9x9_agents3_lasers2-ymo-workstation").

Usage:
    # Resolve a file `git` left with unresolved conflict markers after a pull/merge.
    python scripts/merge_tuning_journal.py tunings/perspective.journal --conflicted tunings/perspective.journal --output /tmp/merged.journal

    # Merge two already-separate journal files (e.g. after manually fetching the other side).
    python scripts/merge_tuning_journal.py --local tunings/perspective.journal --remote /tmp/theirs.journal --output /tmp/merged.journal
"""

import argparse
import re
import socket
import tempfile
from pathlib import Path

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial

StudyData = dict[str, tuple[FrozenStudy, list[FrozenTrial]]]

_CONFLICT_START = re.compile(rb"^<{7} ")
_CONFLICT_SEP = re.compile(rb"^={7}$")
_CONFLICT_END = re.compile(rb"^>{7} ")


def read_studies(path: Path) -> StudyData:
    """
    Replay a journal file with Optuna's own storage backend and collect every study with its trials.

    @ai-generated
    """
    storage = JournalStorage(JournalFileBackend(str(path)))
    studies: StudyData = {}
    for study in storage.get_all_studies():
        trials = storage.get_all_trials(study._study_id, deepcopy=True)
        studies[study.study_name] = (study, trials)
    return studies


def split_conflict(path: Path) -> tuple[Path, Path]:
    """
    Split a journal file left with unresolved git conflict markers into its "ours" and "theirs"
    sides, each written to its own temporary, independently replayable journal file.

    @ai-generated
    """
    ours: list[bytes] = []
    theirs: list[bytes] = []
    in_conflict = False
    on_theirs_side = False
    with open(path, "rb") as f:
        for line in f:
            if _CONFLICT_START.match(line):
                in_conflict = True
                on_theirs_side = False
                continue
            if in_conflict and _CONFLICT_SEP.match(line):
                on_theirs_side = True
                continue
            if in_conflict and _CONFLICT_END.match(line):
                in_conflict = False
                on_theirs_side = False
                continue
            if not in_conflict:
                ours.append(line)
                theirs.append(line)
            elif on_theirs_side:
                theirs.append(line)
            else:
                ours.append(line)

    ours_path = Path(tempfile.mkstemp(suffix="-ours.journal")[1])
    theirs_path = Path(tempfile.mkstemp(suffix="-theirs.journal")[1])
    ours_path.write_bytes(b"".join(ours))
    theirs_path.write_bytes(b"".join(theirs))
    return ours_path, theirs_path


def merge_studies(canonical: StudyData, duplicate: StudyData, duplicate_label: str) -> StudyData:
    """
    Combine two sets of studies, keeping `canonical` untouched and renaming any study in
    `duplicate` that shares a name with one in `canonical`, so trials from two independently-run
    studies are never mixed together.

    @ai-generated
    """
    merged = dict(canonical)
    for name, entry in duplicate.items():
        target_name = name
        if target_name in merged:
            target_name = f"{name}-{duplicate_label}"
            if target_name in merged:
                raise SystemExit(f"Cannot disambiguate duplicate study {name!r}: {target_name!r} already exists too.")
            print(f"Study name collision on {name!r}: keeping both, local copy renamed to {target_name!r}.")
        merged[target_name] = entry
    return merged


def write_journal(output_path: Path, studies: StudyData) -> None:
    """
    Write a fresh journal file containing every given study and trial, preserving each trial's
    original parameters, state, values, timestamps and attributes instead of re-running anything.

    @ai-generated
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        raise SystemExit(f"Refusing to write into a non-empty file: {output_path}")

    storage = JournalStorage(JournalFileBackend(str(output_path)))
    for name, (study, trials) in studies.items():
        study_id = storage.create_new_study(study.directions, study_name=name)
        for key, value in study.user_attrs.items():
            storage.set_study_user_attr(study_id, key, value)
        for key, value in study.system_attrs.items():
            storage.set_study_system_attr(study_id, key, value)
        for trial in trials:
            storage.create_new_trial(study_id, template_trial=trial)
        print(f"Wrote study {name!r} with {len(trials)} trial(s).")


def main() -> None:
    """
    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Path of the merged journal file to create (must not already exist/be non-empty).")
    parser.add_argument("--conflicted", type=Path, help="A single journal file still containing unresolved git conflict markers.")
    parser.add_argument("--local", type=Path, help="This machine's journal file (used together with --remote).")
    parser.add_argument("--remote", type=Path, help="The other workstation's journal file (used together with --local).")
    parser.add_argument(
        "--hostname",
        default=socket.gethostname(),
        help="Label appended to this machine's studies when a name collides with the remote side (default: this machine's hostname).",
    )
    args = parser.parse_args()

    if args.conflicted and (args.local or args.remote):
        parser.error("--conflicted cannot be combined with --local/--remote")
    if bool(args.local) != bool(args.remote):
        parser.error("--local and --remote must be given together")
    if not args.conflicted and not args.local:
        parser.error("either --conflicted or --local/--remote is required")

    cleanup: list[Path] = []
    try:
        if args.conflicted:
            local_path, remote_path = split_conflict(args.conflicted)
            cleanup += [local_path, remote_path]
        else:
            local_path, remote_path = args.local, args.remote

        local_studies = read_studies(local_path)
        remote_studies = read_studies(remote_path)
        merged = merge_studies(remote_studies, local_studies, args.hostname)
        write_journal(args.output, merged)
    finally:
        for p in cleanup:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
