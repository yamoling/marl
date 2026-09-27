"""Persistent, local-only workspace metadata for Studio."""

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

from .errors import bad_request, not_found


class Workspaces:
    def __init__(self, path: Path, default_root: Path, on_change=None):
        """Load workspace roots, migrating name-only and legacy settings to the server default. @ai-edited"""
        self.path = path
        self.default_root = default_root.resolve()
        self.on_change = on_change
        self.lock = threading.RLock()
        if path.exists():
            data = json.loads(path.read_text())
            self.items = {
                key: {"name": item["name"], "logdir": str(Path(item.get("logdir", self.default_root)).expanduser().resolve())}
                for key, item in data["workspaces"].items()
            }
            self.selected = data["selected"]
            if self.selected is not None and self.selected not in self.items:
                raise ValueError("Selected workspace does not exist")
        else:
            identifier = uuid.uuid4().hex
            self.items = {identifier: {"name": "Default", "logdir": str(self.default_root)}}
            self.selected = identifier

    def listing(self) -> dict:
        """Return an independent snapshot of names and the selected ID. @ai-edited"""
        with self.lock:
            return {"selected": self.selected, "workspaces": [self._item(key) for key in self.items]}

    def _item(self, key: str) -> dict:
        return {"id": key, **self.items[key]}

    def _save(self):
        """Atomically replace the configuration; never write inside an experiment. @ai-generated"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".studio-workspaces-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump({"selected": self.selected, "workspaces": self.items}, stream)
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def _require(self, key: str):
        if key not in self.items:
            raise not_found(f"Unknown workspace {key}", "unknown-workspace")

    def create(self, name: str, logdir: str | None = None) -> dict:
        """Create a workspace with its own root, defaulting to the current server root. @ai-edited"""
        name = self._name(name)
        with self.lock:
            root = (
                self._root(logdir)
                if logdir is not None
                else Path(self.items[self.selected]["logdir"])
                if self.selected
                else self.default_root
            )
            key = uuid.uuid4().hex
            self.items[key] = {"name": name, "logdir": str(root)}
            self._save()
            return self._item(key)

    def rename(self, key: str, name: str) -> dict:
        """Rename a workspace without changing its selection. @ai-edited"""
        name = self._name(name)
        with self.lock:
            self._require(key)
            self.items[key]["name"] = name
            self._save()
            return self._item(key)

    def set_logdir(self, key: str, logdir: str) -> dict:
        """Change a workspace root and refresh the library if it is selected. @ai-generated"""
        root = self._root(logdir)
        with self.lock:
            self._require(key)
            if self.items[key]["logdir"] != str(root):
                self.items[key]["logdir"] = str(root)
                self._save()
                if self.selected == key and self.on_change is not None:
                    self.on_change(root)
            return self._item(key)

    def delete(self, key: str) -> dict:
        """Remove workspace metadata, selecting another if needed. @ai-edited"""
        with self.lock:
            self._require(key)
            was_selected = self.selected == key
            del self.items[key]
            if was_selected:
                self.selected = next(iter(self.items), None)
            self._save()
            if was_selected and self.selected is not None and self.on_change is not None:
                self.on_change(Path(self.items[self.selected]["logdir"]))
            return self.listing()

    def select(self, key: str) -> dict:
        """Select a workspace and switch the shared library to its root. @ai-edited"""
        with self.lock:
            self._require(key)
            if self.selected != key:
                root = self._root(self.items[key]["logdir"])
                self.selected = key
                self._save()
                if self.on_change is not None:
                    self.on_change(root)
            return self._item(key)

    @staticmethod
    def _root(value: str) -> Path:
        """Require an existing directory before making it a selectable logs root. @ai-generated"""
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise bad_request("logdir must be a nonempty directory path")
        root = Path(value.strip()).expanduser().resolve()
        if not root.is_dir():
            raise bad_request("logdir must be an existing directory")
        return root

    @staticmethod
    def _name(name: str) -> str:
        """Require a meaningful display name. @ai-generated"""
        if not isinstance(name, str) or not name.strip():
            raise bad_request("name must be a nonempty string")
        return name.strip()
