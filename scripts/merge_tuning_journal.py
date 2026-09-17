"""
Merge Optuna journal files produced independently by two workstations tuning in parallel.

The journal is append-only and both machines write to their own copy of the same file, so `git`
ends up with a conflict once both sides are pushed. This script never re-runs any trial. It copies
the local journal byte-for-byte, then appends the remote records after rebasing their numeric study
and trial IDs. Keeping the existing records untouched makes the resulting git diff contain only the
records that were actually merged.

Studies are expected to have distinct names across machines (see `scripts/tuning.py`, which derives
the name from the algorithm and layout). If the same study name shows up on both sides, its trials
are *not* merged together: the local (this machine's) copy is kept as a separate study, renamed by
appending `-<hostname>` (e.g. "DQN-interdependent-2-9x9_agents3_lasers2-ymo-workstation").

Usage:
    # Resolve a file `git` left with unresolved conflict markers after a pull/merge.
    python scripts/merge_tuning_journal.py --conflicted tunings/perspective.journal --output /tmp/merged.journal

    # Merge two already-separate journal files (e.g. after manually fetching the other side).
    python scripts/merge_tuning_journal.py --local tunings/perspective.journal --remote /tmp/theirs.journal --output /tmp/merged.journal
"""

import argparse
import json
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
_STUDY_ID = re.compile(rb'("study_id"\s*:\s*)(\d+)')
_TRIAL_ID = re.compile(rb'("trial_id"\s*:\s*)(\d+)')


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


def local_study_renames(local: StudyData, remote: StudyData, local_label: str) -> dict[str, str]:
    """
    Return the local study-name changes needed to keep the remote names canonical when the two
    journals contain independently-run studies with the same name.

    @ai-generated
    """
    occupied_names = set(remote)
    renames: dict[str, str] = {}
    for name in local:
        target_name = name
        if target_name in occupied_names:
            target_name = f"{name}-{local_label}"
            if target_name in occupied_names:
                raise SystemExit(f"Cannot disambiguate duplicate study {name!r}: {target_name!r} already exists too.")
            print(f"Study name collision on {name!r}: keeping both, local copy renamed to {target_name!r}.")
            renames[name] = target_name
        occupied_names.add(target_name)
    return renames


def journal_id_offsets(path: Path) -> tuple[int, int]:
    """
    Count study and trial creation records, which are the offsets Optuna will use when replaying
    records appended from another independently-created journal.

    @ai-generated
    """
    study_count = 0
    trial_count = 0
    with open(path, "rb") as journal:
        for line in journal:
            op_code = json.loads(line)["op_code"]
            study_count += op_code == 0
            trial_count += op_code == 4
    return study_count, trial_count


def rename_study(line: bytes, old_name: str, new_name: str) -> bytes:
    """Rename one study creation record without reserializing any other part of the line. @ai-generated"""
    old_field = b'"study_name":' + json.dumps(old_name, ensure_ascii=False).encode()
    new_field = b'"study_name":' + json.dumps(new_name, ensure_ascii=False).encode()
    if old_field not in line:
        return line
    return line.replace(old_field, new_field, 1)


def rebase_record_ids(line: bytes, study_offset: int, trial_offset: int) -> bytes:
    """
    Shift IDs in one remote journal record while leaving its remaining serialized bytes untouched.

    @ai-generated
    """
    line = _STUDY_ID.sub(lambda match: match[1] + str(int(match[2]) + study_offset).encode(), line)
    return _TRIAL_ID.sub(lambda match: match[1] + str(int(match[2]) + trial_offset).encode(), line)


def write_journal(output_path: Path, local_path: Path, remote_path: Path, renames: dict[str, str]) -> None:
    """
    Copy the local journal with minimal study-name edits, then append the remote journal with only
    its numeric IDs changed. This deliberately avoids parsing and reserializing complete records.

    @ai-edited
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        raise SystemExit(f"Refusing to write into a non-empty file: {output_path}")

    study_offset, trial_offset = journal_id_offsets(local_path)
    with open(output_path, "wb") as output:
        local_ends_with_newline = True
        with open(local_path, "rb") as local:
            for line in local:
                for old_name, new_name in renames.items():
                    line = rename_study(line, old_name, new_name)
                output.write(line)
                local_ends_with_newline = line.endswith(b"\n")

        if output.tell() and not local_ends_with_newline:
            output.write(b"\n")

        with open(remote_path, "rb") as remote:
            for line in remote:
                output.write(rebase_record_ids(line, study_offset, trial_offset))


def main() -> None:
    """Merge the requested journal inputs into a fresh, minimally changed output. @ai-edited"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path, required=True, help="Path of the merged journal file to create (must not already exist/be non-empty)."
    )
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
        renames = local_study_renames(local_studies, remote_studies, args.hostname)
        write_journal(args.output, local_path, remote_path, renames)

        merged_studies = read_studies(args.output)
        expected_trials = sum(len(trials) for _, trials in local_studies.values()) + sum(
            len(trials) for _, trials in remote_studies.values()
        )
        actual_trials = sum(len(trials) for _, trials in merged_studies.values())
        if len(merged_studies) != len(local_studies) + len(remote_studies) or actual_trials != expected_trials:
            raise SystemExit("Merged journal failed validation; the input files were left untouched.")
        print(f"Wrote {len(merged_studies)} studies with {actual_trials} trials to {args.output}.")
    finally:
        for p in cleanup:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
