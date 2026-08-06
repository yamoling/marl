#!/usr/bin/env python3
"""Synchronize project directories with Nextcloud over WebDAV."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import threading
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote

from dotenv import load_dotenv
from tqdm import tqdm
from webdav3.client import Client
from webdav3.exceptions import WebDavException

NEXTCLOUD_URL = "https://mycloud.ulb.be"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "sync.toml"
CACHE_PATH = PROJECT_ROOT / ".nextcloud-sync-cache.json"
CACHE_VERSION = 1
Direction = Literal["full", "up", "down"]
TransferAction = Literal["upload", "download"]
MAX_TRANSFER_ATTEMPTS = 5


@dataclass(frozen=True)
class RemoteFile:
    """Metadata needed to compare and download a remote file."""

    path: str
    modified: float
    size: int


@dataclass(frozen=True)
class TransferFailure:
    """A file transfer that was skipped after exhausting its retries."""

    action: TransferAction
    path: str
    attempts: int
    error: str


@dataclass(frozen=True)
class TransferTask:
    """A file transfer ready to run in the worker pool."""

    action: TransferAction
    local_path: Path
    remote_file: RemoteFile


@dataclass(frozen=True)
class DirectoryPlan:
    """Planned transfers and directory creation for one synchronized tree."""

    directory: PurePosixPath
    tasks: list[TransferTask]
    remote_directories: set[str]
    local_directories: set[Path]


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer.

    @ai-generated
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_arguments() -> argparse.Namespace:
    """Parse directories and the optional one-way synchronization direction.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description="Synchronize project directories with Nextcloud.")
    parser.add_argument(
        "directories",
        nargs="*",
        metavar="DIRECTORY",
        help="project-relative directories (defaults to those in sync.toml)",
    )
    parser.add_argument(
        "-d",
        "--direction",
        choices=("up", "down"),
        help="force upload or download; omit for newest-file-wins full sync",
    )
    parser.add_argument(
        "--n-workers",
        type=positive_integer,
        default=1,
        help="number of simultaneous file transfers (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the planned transfer counts and sizes without changing files",
    )
    return parser.parse_args()


def load_configured_directories(config_path: Path = CONFIG_PATH) -> list[str]:
    """Load the top-level ``directories`` list from ``sync.toml``.

    @ai-generated
    """
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ValueError(f"configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {config_path}: {error}") from error

    directories = config.get("directories")
    if not isinstance(directories, list) or not directories:
        raise ValueError(f"{config_path} must define a non-empty 'directories' list")
    if any(not isinstance(directory, str) or not directory.strip() for directory in directories):
        raise ValueError("every entry in 'directories' must be a non-empty string")
    return directories


def load_ignore_patterns(config_path: Path = CONFIG_PATH) -> list[str]:
    """Load the optional top-level ``ignore`` glob pattern list from ``sync.toml``.

    @ai-generated
    """
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ValueError(f"configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {config_path}: {error}") from error

    patterns = config.get("ignore", [])
    if not isinstance(patterns, list) or any(
        not isinstance(pattern, str) or not pattern.strip() for pattern in patterns
    ):
        raise ValueError(f"'ignore' in {config_path} must be a list of non-empty strings")
    return patterns


def is_ignored(path_parts: tuple[str, ...], patterns: list[str]) -> bool:
    """Check whether any path segment matches an ignore glob pattern.

    @ai-generated
    """
    return any(fnmatch.fnmatch(part, pattern) for part in path_parts for pattern in patterns)


def normalize_directories(directories: list[str]) -> list[PurePosixPath]:
    """Validate, normalize, and deduplicate project-relative directory names.

    @ai-generated
    """
    normalized: list[PurePosixPath] = []
    seen: set[str] = set()
    for raw_directory in directories:
        directory = PurePosixPath(raw_directory.replace("\\", "/"))
        if directory.is_absolute() or not directory.parts or ".." in directory.parts:
            raise ValueError(f"directory must stay inside the project: {raw_directory!r}")
        clean = PurePosixPath(*[part for part in directory.parts if part not in ("", ".")])
        if not clean.parts:
            raise ValueError("the project root cannot be synchronized as a directory")
        name = clean.as_posix()
        if name not in seen:
            seen.add(name)
            normalized.append(clean)
    return normalized


def load_credentials() -> tuple[str, str]:
    """Load the required Nextcloud credentials from the project ``.env`` file.

    @ai-generated
    """
    load_dotenv(PROJECT_ROOT / ".env")
    username = os.environ.get("NEXTCLOUD_USERNAME")
    password = os.environ.get("NEXTCLOUD_PASSWORD")
    if not username or not password:
        raise ValueError("NEXTCLOUD_USERNAME and NEXTCLOUD_PASSWORD must be set in the project .env")
    return username, password


def format_size(size: int) -> str:
    """Format a byte count using compact binary units.

    @ai-generated
    """
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def print_overview(tasks: list[TransferTask]) -> None:
    """Print aggregate counts and sizes for planned transfers.

    @ai-generated
    """
    uploads = [task for task in tasks if task.action == "upload"]
    downloads = [task for task in tasks if task.action == "download"]
    print("Synchronization overview:")
    print(f"  Files to upload:    {len(uploads)}")
    print(f"  Files to download:  {len(downloads)}")
    print(f"  Total upload size:  {format_size(sum(task.remote_file.size for task in uploads))}")
    print(f"  Total download size: {format_size(sum(task.remote_file.size for task in downloads))}")


def parse_remote_mtime(value: Any) -> float:
    """Convert a WebDAV HTTP modification date to a Unix timestamp.

    @ai-generated
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"Nextcloud returned an invalid modification time: {value!r}")
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot parse remote modification time {value!r}") from error


def parse_remote_size(value: Any) -> int:
    """Convert WebDAV file-size metadata to a non-negative byte count.

    @ai-generated
    """
    try:
        size = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Nextcloud returned an invalid file size: {value!r}") from error
    if size < 0:
        raise ValueError(f"Nextcloud returned a negative file size: {size}")
    return size


class NextcloudSynchronizer:
    """Synchronize project-relative directory trees with a Nextcloud account."""

    def __init__(
        self,
        username: str,
        password: str,
        ignore_patterns: list[str] | None = None,
        n_workers: int = 1,
    ) -> None:
        options = {
            "webdav_hostname": (f"{NEXTCLOUD_URL}/remote.php/dav/files/{username}/"),
            "webdav_login": username,
            "webdav_password": password,
        }
        self.client = Client(options)
        self.n_workers = n_workers
        self.ignore_patterns = ignore_patterns or []
        self.failures: list[TransferFailure] = []
        self._client_options = options
        self._thread_local = threading.local()
        self._failure_lock = threading.Lock()
        self._cache = self.load_inventory_cache(username, self.ignore_patterns)
        self._cache_dirty = False

    def authenticate(self) -> None:
        """Verify that the configured WebDAV account can list its root directory."""
        self.client.list("/")

    @staticmethod
    def load_inventory_cache(username: str, ignore_patterns: list[str]) -> dict[str, Any]:
        """Load a compatible remote inventory cache or create an empty one.

        The cache is invalidated whenever the configured ignore patterns change,
        since cached entries were filtered using the previous patterns.

        @ai-generated
        """
        empty_cache: dict[str, Any] = {
            "version": CACHE_VERSION,
            "server": NEXTCLOUD_URL,
            "username": username,
            "ignore_patterns": sorted(ignore_patterns),
            "directories": {},
        }
        try:
            with CACHE_PATH.open(encoding="utf-8") as cache_file:
                cache = json.load(cache_file)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return empty_cache

        if not isinstance(cache, dict):
            return empty_cache
        if (
            cache.get("version") != CACHE_VERSION
            or cache.get("server") != NEXTCLOUD_URL
            or cache.get("username") != username
            or cache.get("ignore_patterns") != sorted(ignore_patterns)
            or not isinstance(cache.get("directories"), dict)
        ):
            return empty_cache
        return cache

    def save_inventory_cache(self) -> None:
        """Atomically persist remote inventories learned during this run.

        @ai-generated
        """
        if not self._cache_dirty:
            return
        temporary_path = CACHE_PATH.with_suffix(".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as cache_file:
                json.dump(self._cache, cache_file, separators=(",", ":"))
            temporary_path.replace(CACHE_PATH)
            self._cache_dirty = False
        except OSError as error:
            tqdm.write(f"warning: could not save inventory cache: {error}", file=sys.stderr)

    def get_transfer_client(self) -> Client:
        """Return a worker-local WebDAV client for thread-safe transfers.

        @ai-generated
        """
        if self.n_workers == 1:
            return self.client
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = Client(self._client_options)
            self._thread_local.client = client
        return client

    def ensure_remote_directories(self, directories: set[str]) -> None:
        """Create missing remote directories in parent-before-child order.

        @ai-generated
        """
        required: set[str] = set()
        for raw_directory in directories:
            directory = PurePosixPath(raw_directory)
            while directory.parts:
                required.add(directory.as_posix())
                directory = directory.parent
                if directory == PurePosixPath("."):
                    break

        for directory in sorted(required, key=lambda path: (path.count("/"), path)):
            if not self.client.check(directory):
                print(f"  create remote directory: {directory}")
                self.client.mkdir(directory)

    def collect_local_tree(
        self,
        root: Path,
    ) -> tuple[dict[str, Path], set[str]]:
        """Collect regular files and subdirectories below a local root.

        Names matching a configured ignore pattern are pruned from the walk, so
        neither they nor anything below them is included.

        @ai-generated
        """
        files: dict[str, Path] = {}
        directories: set[str] = set()
        if not root.exists():
            return files, directories
        if not root.is_dir():
            raise ValueError(f"local synchronization path is not a directory: {root}")

        for current_root, directory_names, file_names in os.walk(root):
            current = Path(current_root)
            directory_names[:] = [
                name
                for name in directory_names
                if not (current / name).is_symlink() and not is_ignored((name,), self.ignore_patterns)
            ]
            for name in directory_names:
                directories.add((current / name).relative_to(root).as_posix())
            for name in file_names:
                if is_ignored((name,), self.ignore_patterns):
                    continue
                path = current / name
                if path.is_file() and not path.is_symlink():
                    files[path.relative_to(root).as_posix()] = path
        return files, directories

    def collect_remote_tree(self, remote_root: str) -> tuple[dict[str, RemoteFile], set[str]]:
        """Collect remote files and subdirectories from a recursive PROPFIND.

        The listing is already flattened by the server, so entries whose relative
        path contains a segment matching a configured ignore pattern are skipped
        individually rather than pruned from a walk.

        @ai-generated
        """
        files: dict[str, RemoteFile] = {}
        directories: set[str] = set()
        marker = f"/{remote_root.strip('/')}/"

        for item in self.client.list(remote_root, get_info=True, recursive=True):
            raw_path = unquote(str(item.get("path", ""))).rstrip("/")
            marker_index = raw_path.find(marker)
            if marker_index < 0:
                continue
            relative_path = raw_path[marker_index + len(marker) :]
            relative = PurePosixPath(relative_path)
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                continue
            if is_ignored(relative.parts, self.ignore_patterns):
                continue

            relative_name = relative.as_posix()
            if item.get("isdir"):
                directories.add(relative_name)
            else:
                files[relative_name] = RemoteFile(
                    path=f"{remote_root}/{relative_name}",
                    modified=parse_remote_mtime(item.get("modified")),
                    size=parse_remote_size(item.get("size")),
                )
        return files, directories

    def deserialize_cached_tree(self, remote_root: str, entry: Any) -> tuple[dict[str, RemoteFile], set[str]] | None:
        """Validate and reconstruct one cached remote inventory.

        @ai-generated
        """
        if not isinstance(entry, dict):
            return None
        raw_files = entry.get("files")
        raw_directories = entry.get("directories")
        if not isinstance(raw_files, dict) or not isinstance(raw_directories, list):
            return None

        files: dict[str, RemoteFile] = {}
        directories: set[str] = set()
        try:
            for relative, metadata in raw_files.items():
                if not isinstance(relative, str) or not isinstance(metadata, dict):
                    return None
                files[relative] = RemoteFile(
                    path=f"{remote_root}/{relative}",
                    modified=float(metadata["modified"]),
                    size=parse_remote_size(metadata["size"]),
                )
            for relative in raw_directories:
                if not isinstance(relative, str):
                    return None
                directories.add(relative)
        except (KeyError, TypeError, ValueError):
            return None
        return files, directories

    def collect_remote_tree_cached(self, remote_root: str) -> tuple[dict[str, RemoteFile], set[str]]:
        """Reuse a remote inventory when its root ETag is unchanged.

        @ai-generated
        """
        remote_info = self.client.info(remote_root)
        etag = remote_info.get("etag")
        cache_directories = self._cache["directories"]
        cached_entry = cache_directories.get(remote_root)
        if isinstance(etag, str) and isinstance(cached_entry, dict) and cached_entry.get("etag") == etag:
            cached_tree = self.deserialize_cached_tree(remote_root, cached_entry)
            if cached_tree is not None:
                print(f"Using cached remote inventory for {remote_root}")
                return cached_tree

        print(f"Refreshing remote inventory for {remote_root}...")
        files, directories = self.collect_remote_tree(remote_root)
        confirmed_etag = self.client.info(remote_root).get("etag")
        if isinstance(etag, str) and confirmed_etag == etag:
            cache_directories[remote_root] = {
                "etag": etag,
                "files": {relative: {"modified": item.modified, "size": item.size} for relative, item in files.items()},
                "directories": sorted(directories),
            }
            self._cache_dirty = True
        return files, directories

    def transfer_with_retries(
        self,
        action: TransferAction,
        display_path: str,
        operation: Callable[[], None],
    ) -> bool:
        """Run one transfer repeatedly and record it if every attempt fails.

        @ai-generated
        """
        last_error: OSError | ValueError | WebDavException | None = None
        for attempt in range(1, MAX_TRANSFER_ATTEMPTS + 1):
            try:
                operation()
                return True
            except (OSError, ValueError, WebDavException) as error:
                last_error = error
                if attempt < MAX_TRANSFER_ATTEMPTS:
                    tqdm.write(
                        f"{action} failed for {display_path} "
                        f"(attempt {attempt}/{MAX_TRANSFER_ATTEMPTS}); retrying: {error}",
                        file=sys.stderr,
                    )

        assert last_error is not None
        with self._failure_lock:
            self.failures.append(
                TransferFailure(
                    action=action,
                    path=display_path,
                    attempts=MAX_TRANSFER_ATTEMPTS,
                    error=str(last_error),
                )
            )
        tqdm.write(
            f"skipping {display_path} after {MAX_TRANSFER_ATTEMPTS} failed {action} attempts",
            file=sys.stderr,
        )
        return False

    def upload(self, local_path: Path, remote_path: str) -> bool:
        """Upload one file with retries and synchronize its local mtime.

        @ai-generated
        """
        display_path = local_path.relative_to(PROJECT_ROOT).as_posix()

        def upload_once() -> None:
            """Perform one upload attempt and retrieve its resulting mtime.

            @ai-generated
            """
            client = self.get_transfer_client()
            client.upload(remote_path, str(local_path))
            remote_mtime = parse_remote_mtime(client.info(remote_path).get("modified"))
            os.utime(local_path, (remote_mtime, remote_mtime))

        return self.transfer_with_retries("upload", display_path, upload_once)

    def download(self, remote_file: RemoteFile, local_path: Path) -> bool:
        """Download one file with retries and preserve its remote mtime.

        @ai-generated
        """
        display_path = local_path.relative_to(PROJECT_ROOT).as_posix()

        def download_once() -> None:
            """Perform one download attempt and apply the remote mtime.

            @ai-generated
            """
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.get_transfer_client().download(remote_file.path, str(local_path))
            os.utime(local_path, (remote_file.modified, remote_file.modified))

        return self.transfer_with_retries("download", display_path, download_once)

    def execute_transfers(self, tasks: list[TransferTask]) -> tuple[int, int]:
        """Execute transfer tasks with the configured degree of parallelism.

        @ai-generated
        """

        def execute(task: TransferTask) -> tuple[TransferAction, bool]:
            """Execute one prepared upload or download task.

            @ai-generated
            """
            if task.action == "upload":
                succeeded = self.upload(task.local_path, task.remote_file.path)
            else:
                succeeded = self.download(task.remote_file, task.local_path)
            return task.action, succeeded

        uploads = 0
        downloads = 0
        with tqdm(total=len(tasks), desc="Synchronizing", unit="file") as progress:
            if self.n_workers == 1:
                for task in tasks:
                    action, succeeded = execute(task)
                    if succeeded and action == "upload":
                        uploads += 1
                    elif succeeded:
                        downloads += 1
                    progress.update()
                return uploads, downloads

            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(execute, task) for task in tasks]
                for future in as_completed(futures):
                    action, succeeded = future.result()
                    if succeeded and action == "upload":
                        uploads += 1
                    elif succeeded:
                        downloads += 1
                    progress.update()
        return uploads, downloads

    def plan_directory(self, directory: PurePosixPath, direction: Direction) -> DirectoryPlan:
        """Plan one directory without changing local or remote data.

        Full mode transfers files missing on either side and resolves conflicts
        using modification times. One-way modes update only differing source files
        and leave files that exist only at the destination untouched.

        @ai-generated
        """
        local_root = PROJECT_ROOT.joinpath(*directory.parts)
        remote_root = directory.as_posix()
        local_exists = local_root.exists()
        remote_exists = self.client.check(remote_root)

        if direction == "up" and not local_exists:
            raise ValueError(f"local directory does not exist: {local_root}")
        if direction == "down" and not remote_exists:
            raise ValueError(f"remote directory does not exist: {remote_root}")
        if direction == "full" and not local_exists and not remote_exists:
            raise ValueError(f"directory does not exist locally or remotely: {remote_root}")
        if local_exists and not local_root.is_dir():
            raise ValueError(f"local synchronization path is not a directory: {local_root}")

        local_files, local_directories = (
            self.collect_local_tree(local_root) if direction in ("up", "down", "full") else ({}, set())
        )
        remote_files, remote_directories = (
            self.collect_remote_tree_cached(remote_root)
            if direction in ("down", "full") and remote_exists
            else ({}, set())
        )

        tasks: list[TransferTask] = []
        for relative in sorted(set(local_files) | set(remote_files)):
            local_path = local_files.get(relative)
            remote_file = remote_files.get(relative)
            remote_path = f"{remote_root}/{relative}"

            if direction == "up":
                if local_path is not None:
                    tasks.append(
                        TransferTask(
                            "upload",
                            local_path,
                            RemoteFile(remote_path, 0, local_path.stat().st_size),
                        )
                    )
                continue
            if direction == "down":
                if remote_file is not None:
                    if local_path is None:
                        tasks.append(TransferTask("download", local_root / relative, remote_file))
                    else:
                        local_stat = local_path.stat()
                        # WebDAV dates have one-second precision, unlike many local filesystems.
                        local_mtime = int(local_stat.st_mtime)
                        remote_mtime = int(remote_file.modified)
                        if local_mtime != remote_mtime or local_stat.st_size != remote_file.size:
                            tasks.append(TransferTask("download", local_path, remote_file))
                continue

            if local_path is None and remote_file is not None:
                tasks.append(TransferTask("download", local_root / relative, remote_file))
            elif remote_file is None and local_path is not None:
                tasks.append(
                    TransferTask(
                        "upload",
                        local_path,
                        RemoteFile(remote_path, 0, local_path.stat().st_size),
                    )
                )
            elif local_path is not None and remote_file is not None:
                local_stat = local_path.stat()
                # WebDAV dates have one-second precision, unlike many local filesystems.
                local_mtime = int(local_stat.st_mtime)
                remote_mtime = int(remote_file.modified)
                if local_mtime > remote_mtime:
                    tasks.append(
                        TransferTask(
                            "upload",
                            local_path,
                            RemoteFile(remote_path, 0, local_stat.st_size),
                        )
                    )
                elif remote_mtime > local_mtime:
                    tasks.append(TransferTask("download", local_path, remote_file))

        required_remote_directories = set()
        if direction in ("up", "full") and local_exists:
            required_remote_directories = {remote_root} | {
                f"{remote_root}/{relative}" for relative in local_directories
            }

        required_local_directories = set()
        if direction in ("down", "full") and remote_exists:
            required_local_directories = {local_root} | {local_root / relative for relative in remote_directories}

        return DirectoryPlan(
            directory=directory,
            tasks=tasks,
            remote_directories=required_remote_directories,
            local_directories=required_local_directories,
        )

    def prepare_directories(self, plans: list[DirectoryPlan]) -> None:
        """Create directories needed by a collection of transfer plans.

        @ai-generated
        """
        remote_directories = {directory for plan in plans for directory in plan.remote_directories}
        self.ensure_remote_directories(remote_directories)
        local_directories = {directory for plan in plans for directory in plan.local_directories}
        for directory in sorted(local_directories, key=lambda path: len(path.parts)):
            directory.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Load configuration and synchronize every selected directory.

    @ai-generated
    """
    args = parse_arguments()
    try:
        requested = args.directories or load_configured_directories()
        directories = normalize_directories(requested)
        ignore_patterns = load_ignore_patterns()
        username, password = load_credentials()
        direction: Direction = args.direction or "full"

        synchronizer = NextcloudSynchronizer(
            username,
            password,
            ignore_patterns=ignore_patterns,
            n_workers=args.n_workers,
        )
        synchronizer.authenticate()

        plans = [synchronizer.plan_directory(directory, direction) for directory in directories]
        tasks = [task for plan in plans for task in plan.tasks]
        print_overview(tasks)
        if args.dry_run:
            print("Dry run complete; no files or directories were changed.")
            return 0
        if len(tasks) == 0:
            return 0
        synchronizer.prepare_directories(plans)
        total_uploads, total_downloads = synchronizer.execute_transfers(tasks)
        synchronizer.save_inventory_cache()

        if not synchronizer.failures:
            print(f"Everything went fine: {total_uploads} uploaded, {total_downloads} downloaded.")
            return 0

        print(
            f"Sync completed with {len(synchronizer.failures)} failed file transfer(s):",
            file=sys.stderr,
        )
        for failure in synchronizer.failures:
            print(
                f"  - failed {failure.action}: {failure.path} "
                f"({failure.attempts} attempts; last error: {failure.error})",
                file=sys.stderr,
            )
        return 1
    except (OSError, ValueError, WebDavException) as error:
        print(f"Sync failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
