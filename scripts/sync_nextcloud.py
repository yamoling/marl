#!/usr/bin/env python3
"""
Nextcloud synchronization script for logs directory.
Provides bidirectional sync between local logs/ and Nextcloud.

Uses WebDAV protocol for synchronization.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from webdav3.client import Client
except ImportError:
    print("Error: webdav3 library not installed.")
    print("Install it with: pip install webdav3")
    sys.exit(1)


class NextcloudSync:
    """Manages bidirectional sync between local logs directory and Nextcloud."""

    def __init__(
        self,
        server_url: str,
        username: str,
        password: str,
        local_dir: str = "logs",
        remote_dir: str = "logs",
        max_workers: int = 8,
    ):
        """Initialize Nextcloud sync.

        Args:
            server_url: Nextcloud server URL (e.g., https://mycloud.ulb.be)
            username: Nextcloud username
            password: Nextcloud app-specific password
            local_dir: Local directory to sync (relative to repo root)
            remote_dir: Remote directory on Nextcloud
            max_workers: Number of concurrent upload/download threads
        """
        self.local_dir = Path(local_dir).resolve()
        self.remote_dir = remote_dir
        self.cache_file = Path(".sync_cache.json")
        self.server_url = server_url.rstrip("/")
        self.username = username
        self.password = password
        self.max_workers = max_workers
        self.client: Optional[Client] = None
        self.sync_cache: Dict = {}
        self._client_options: Optional[Dict] = None
        self._thread_local = threading.local()
        self._verified_dirs: set = set()
        self._cache_lock = threading.Lock()

    def authenticate(self) -> bool:
        """Authenticate with Nextcloud using WebDAV.

        Returns:
            True if authenticated successfully, False otherwise.
        """
        try:
            # WebDAV URL for Nextcloud: /remote.php/dav/files/{username}/
            webdav_url = f"{self.server_url}/remote.php/dav/files/{self.username}/"

            options = {
                "webdav_hostname": webdav_url,
                "webdav_login": self.username,
                "webdav_password": self.password,
            }

            self._client_options = options
            self.client = Client(options)
            # Test connection
            self.client.list("/")
            print("✓ Successfully authenticated with Nextcloud")

            # Check and create remote directory if needed
            self._ensure_remote_dir(self.remote_dir)

            return True

        except Exception as e:
            print(f"✗ Authentication error: {e}")
            return False

    def _get_thread_client(self) -> Client:
        """Return a WebDAV client private to the calling thread.

        requests.Session (used internally by webdav3) is not safe to share
        across threads under concurrent load, so each worker thread gets its
        own Client built from the same connection options.
        """
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = Client(self._client_options)
            self._thread_local.client = client
        return client

    def _load_cache(self) -> Dict:
        """Load sync cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save sync cache to file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.sync_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _ensure_remote_dir(self, remote_path: str) -> bool:
        """Create remote directory (and any missing parents) if it doesn't exist.

        WebDAV's MKCOL only creates a single level and fails if the parent
        doesn't exist yet, so parents are created first, top-down.

        Args:
            remote_path: Path to directory

        Returns:
            True if directory exists or was created, False otherwise
        """
        remote_path = remote_path.strip("/")
        if not remote_path or remote_path == ".":
            return True

        # Already verified/created earlier in this run - skip the network round-trip
        if remote_path in self._verified_dirs:
            return True

        try:
            if self.client.check(remote_path):
                self._verified_dirs.add(remote_path)
                return True
        except Exception as e:
            print(f"  ✗ Failed to check directory {remote_path}: {e}")
            return False

        # Ensure parent exists first
        parent = str(Path(remote_path).parent)
        if not self._ensure_remote_dir(parent):
            return False

        try:
            print(f"  Creating remote directory: {remote_path}")
            self.client.mkdir(remote_path)
            self._verified_dirs.add(remote_path)
            return True
        except Exception as e:
            print(f"  ✗ Failed to create directory {remote_path}: {e}")
            return False

    @staticmethod
    def _is_test_path(rel_path: Path) -> bool:
        """Check whether any component of a relative path is named 'test'."""
        return "test" in rel_path.parts

    def _ensure_remote_dirs_bulk(self, remote_paths) -> bool:
        """Create many remote directories at once, level by level in parallel.

        WebDAV's MKCOL requires the parent to already exist, so directories
        are grouped by depth and created shallowest-first; siblings within a
        depth level are independent and safe to check/create concurrently.
        With trees of tens of thousands of directories, doing this one
        network round-trip at a time (the naive approach) would take hours.
        """
        all_dirs = set()
        for p in remote_paths:
            p = p.strip("/")
            while p and p != "." and p not in all_dirs:
                all_dirs.add(p)
                p = str(Path(p).parent)

        by_depth: Dict[int, list] = {}
        for d in all_dirs:
            if d in self._verified_dirs:
                continue
            by_depth.setdefault(d.count("/"), []).append(d)

        if not by_depth:
            return True

        total = sum(len(v) for v in by_depth.values())
        done = 0
        dir_workers = max(self.max_workers, 32)  # metadata calls are cheap/latency-bound

        def _ensure_one(d):
            client = self._get_thread_client()
            if not client.check(d):
                client.mkdir(d)
            return d

        print(f"  Verifying/creating {total} remote director{'y' if total == 1 else 'ies'}...")
        for depth in sorted(by_depth):
            with ThreadPoolExecutor(max_workers=dir_workers) as executor:
                futures = {executor.submit(_ensure_one, d): d for d in by_depth[depth]}
                for future in as_completed(futures):
                    d = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  ✗ Failed to create directory {d}: {e}")
                        return False
                    self._verified_dirs.add(d)
                    done += 1
                    if done % 500 == 0 or done == total:
                        print(f"    ...{done}/{total} directories ready")

        return True

    def _find_changed_local_files(self) -> Optional[list]:
        """Scan local_dir and return files that need uploading.

        "test" (sub-)directories are pruned from the walk itself (via
        os.walk, not rglob) so their contents are never even stat'd, rather
        than being scanned and discarded afterwards.

        Change detection uses mtime+size only (the same quick check rsync
        uses by default), not content hashing: at hundreds of thousands of
        files / hundreds of GB, hashing everything just to decide what
        changed would mean reading the whole tree from disk twice - once to
        check, once to actually upload.
        Returns None on a critical error (e.g. missing directory).
        """
        if not self.local_dir.exists():
            print(f"✗ Local directory does not exist: {self.local_dir}")
            return None

        to_upload = []
        scanned = 0
        for root, dirnames, filenames in os.walk(self.local_dir):
            dirnames[:] = [d for d in dirnames if d != "test"]
            root_path = Path(root)

            for filename in filenames:
                if filename == "test":
                    continue

                local_file = root_path / filename
                scanned += 1
                if scanned % 20000 == 0:
                    print(f"  ...scanned {scanned} files")

                rel_path = local_file.relative_to(self.local_dir)
                cache_key = str(rel_path)
                stat = local_file.stat()
                cached = self.sync_cache.get(cache_key, {})

                if cached.get("mtime") == stat.st_mtime and cached.get("size") == stat.st_size:
                    continue

                to_upload.append((local_file, rel_path, stat))

        print(f"  Scanned {scanned} files, {len(to_upload)} need uploading")
        return to_upload

    def upload_local_changes(self) -> int:
        """Upload new/modified local files to Nextcloud, in parallel.

        Returns:
            Number of files uploaded, or -1 on critical error.
        """
        to_upload = self._find_changed_local_files()
        if to_upload is None:
            return -1
        if not to_upload:
            return 0

        needed_dirs = {str(Path(f"{self.remote_dir}/{rel_path.as_posix()}").parent) for _, rel_path, _ in to_upload}
        if not self._ensure_remote_dirs_bulk(needed_dirs):
            print("✗ Could not create required remote directories")
            return -1

        def _upload_one(item):
            local_file, rel_path, stat = item
            remote_path = f"{self.remote_dir}/{rel_path.as_posix()}"
            client = self._get_thread_client()
            client.upload(remote_path, str(local_file))
            return item

        total = len(to_upload)
        uploaded = 0
        print(f"  Uploading {total} file(s)...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_upload_one, item): item for item in to_upload}
            for future in as_completed(futures):
                local_file, rel_path, stat = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ✗ Failed to upload {rel_path}: {e}")
                    for pending in futures:
                        pending.cancel()
                    return -1

                with self._cache_lock:
                    self.sync_cache[str(rel_path)] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "last_sync": datetime.now().isoformat(),
                    }
                uploaded += 1
                print(f"    ...{uploaded}/{total} uploaded (last: {rel_path})")

        print(f"{uploaded}/{total} uploaded")
        return uploaded

    def _find_changed_remote_files(self) -> Optional[list]:
        """List remote_dir recursively and return files that need downloading.

        Uses remote size+modified-time (from the PROPFIND response) to skip
        files that are already up to date, avoiding a wasted download.
        Returns None on a critical error.
        """
        if not self.client.check(self.remote_dir):
            print(f"✗ Remote directory not found: {self.remote_dir}")
            return None

        try:
            # recursive=True is required to descend into subdirectories -
            # without it only the immediate children of remote_dir are listed.
            items = self.client.list(self.remote_dir, get_info=True, recursive=True)
        except Exception as e:
            print(f"✗ Error listing remote folder: {e}")
            return None

        prefix = f"/{self.remote_dir.strip('/')}/"
        to_download = []
        for item in items:
            if item.get("isdir"):
                continue

            # `path` is the full href path returned by the server (includes
            # the WebDAV base path) - pull out everything after remote_dir.
            path = item.get("path", "")
            idx = path.find(prefix)
            if idx == -1:
                continue
            rel_path = path[idx + len(prefix) :]
            if not rel_path:
                continue

            rel_path_obj = Path(rel_path)
            if self._is_test_path(rel_path_obj):
                continue

            remote_size = item.get("size")
            remote_modified = item.get("modified")
            cached = self.sync_cache.get(rel_path, {})
            if cached.get("remote_size") == remote_size and cached.get("remote_modified") == remote_modified:
                continue

            remote_path = f"{self.remote_dir}/{rel_path}"
            to_download.append((remote_path, rel_path, remote_size, remote_modified))

        return to_download

    def download_remote_changes(self) -> int:
        """Download new/modified remote files from Nextcloud, in parallel.

        Returns:
            Number of files downloaded, or -1 on critical error.
        """
        if not self.client:
            print("✗ Not connected to Nextcloud")
            return -1

        self.local_dir.mkdir(parents=True, exist_ok=True)

        to_download = self._find_changed_remote_files()
        if to_download is None:
            return -1
        if not to_download:
            return 0

        # Pre-create local parent directories up front (cheap, local disk only)
        for _, rel_path, _, _ in to_download:
            (self.local_dir / rel_path).parent.mkdir(parents=True, exist_ok=True)

        def _download_one(item):
            remote_path, rel_path, _, _ = item
            client = self._get_thread_client()
            local_file = self.local_dir / rel_path
            client.download(remote_path, str(local_file))
            return item

        total = len(to_download)
        downloaded = 0
        print(f"  Downloading {total} file(s)...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_download_one, item): item for item in to_download}
            for future in as_completed(futures):
                _, rel_path, remote_size, remote_modified = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ✗ Failed to download {rel_path}: {e}")
                    for pending in futures:
                        pending.cancel()
                    return -1

                with self._cache_lock:
                    self.sync_cache[rel_path] = {
                        "remote_size": remote_size,
                        "remote_modified": remote_modified,
                        "last_sync": datetime.now().isoformat(),
                    }
                downloaded += 1
                if downloaded % 100 == 0 or downloaded == total:
                    print(f"    ...{downloaded}/{total} downloaded (last: {rel_path})")

        return downloaded

    def sync(self, direction: str = "both") -> None:
        """Perform synchronization.

        Args:
            direction: 'up' (upload only), 'down' (download only), or 'both' (bidirectional)
        """
        if not self.client:
            if not self.authenticate():
                print("✗ Failed to authenticate with Nextcloud. Exiting.")
                sys.exit(1)

        # Verify remote directory exists
        if not self.client.check(self.remote_dir):
            print(f"✗ Remote directory does not exist: {self.remote_dir}")
            print("  Attempting to create it...")
            if not self._ensure_remote_dir(self.remote_dir):
                print("✗ Failed to create remote directory. Exiting.")
                sys.exit(1)

        self.sync_cache = self._load_cache()

        print(f"\n{'=' * 60}")
        print(f"Nextcloud Sync - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")
        print(f"Server: {self.server_url}")
        print(f"Local:  {self.local_dir}")
        print(f"Remote: {self.remote_dir}")
        print(f"Mode:   {direction}\n")

        uploaded = 0
        downloaded = 0

        if direction in ("up", "both"):
            print("Uploading local changes...")
            uploaded = self.upload_local_changes()
            if uploaded == -1:  # Error occurred
                print("✗ Upload failed. Exiting.")
                sys.exit(1)
            print(f"✓ Uploaded {uploaded} file(s)\n")

        if direction in ("down", "both"):
            print("Downloading remote changes...")
            downloaded = self.download_remote_changes()
            if downloaded == -1:  # Error occurred
                print("✗ Download failed. Exiting.")
                sys.exit(1)
            print(f"✓ Downloaded {downloaded} file(s)\n")

        self._save_cache()

        print(f"{'=' * 60}")
        print(f"Sync complete: {uploaded} uploaded, {downloaded} downloaded")
        print(f"{'=' * 60}\n")


def load_config() -> Optional[Dict]:
    """Load Nextcloud credentials from the environment (.env file).

    Returns:
        Configuration dictionary, or None if any required variable is missing.
    """
    from dotenv import load_dotenv

    load_dotenv()

    server_url = os.environ.get("NEXTCLOUD_URL")
    username = os.environ.get("NEXTCLOUD_USERNAME")
    password = os.environ.get("NEXTCLOUD_PASSWORD")

    if not (server_url and username and password):
        return None

    return {"server_url": server_url, "username": username, "password": password}


def main():
    """Main entry point."""
    import argparse

    # Force line-buffered stdout so progress prints show up immediately even
    # when not attached to a terminal (Python fully buffers stdout otherwise).
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Synchronize logs directory with Nextcloud")
    parser.add_argument("--direction", choices=["up", "down", "both"], default="both", help="Sync direction (default: both)")
    parser.add_argument("--local", default="logs", help="Local directory path (default: logs)")
    parser.add_argument("--remote", default="logs", help="Remote directory path on Nextcloud (default: logs)")
    parser.add_argument("--check-remote", action="store_true", help="Check remote directory status and exit")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent upload/download threads (default: 8)")

    args = parser.parse_args()

    # Load credentials from .env
    config = load_config()
    if not config:
        print("Error: Missing Nextcloud credentials in .env")
        print("\nAdd the following to your .env file:")
        print('NEXTCLOUD_URL="https://mycloud.ulb.be"')
        print('NEXTCLOUD_USERNAME="your_username"')
        print('NEXTCLOUD_PASSWORD="your_app_password"')
        sys.exit(1)

    # Initialize syncer
    syncer = NextcloudSync(
        server_url=config["server_url"],
        username=config["username"],
        password=config["password"],
        local_dir=args.local,
        remote_dir=args.remote,
        max_workers=args.workers,
    )

    # Check remote directory if requested
    if args.check_remote:
        print(f"Checking remote directory: {args.remote}")
        if syncer.authenticate():
            if syncer.client.check(args.remote):
                print(f"✓ Remote directory exists: {args.remote}")
                try:
                    items = syncer.client.list(args.remote, get_info=True)
                    print(f"  Contents ({len(items)} items):")
                    for item in items[:10]:  # Show first 10 items
                        print(f"    - {item}")
                    if len(items) > 10:
                        print(f"    ... and {len(items) - 10} more items")
                except Exception as e:
                    print(f"  Error listing contents: {e}")
            else:
                print(f"✗ Remote directory does NOT exist: {args.remote}")
                print("  Creating directory...")
                syncer._ensure_remote_dir(args.remote)
        else:
            print("✗ Authentication failed")
        sys.exit(0)

    # Run sync
    syncer.sync(direction=args.direction)


if __name__ == "__main__":
    main()
