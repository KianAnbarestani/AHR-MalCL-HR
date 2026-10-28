"""Best-effort exclusive-create run lease; Drive semantics remain non-transactional."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any


class LeaseError(RuntimeError):
    pass


class RunLease:
    def __init__(self, path: str | Path, *, ttl_seconds: float = 900, owner_token: str | None = None):
        if ttl_seconds <= 0:
            raise ValueError("lease TTL must be positive")
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self.owner_token = owner_token or uuid.uuid4().hex
        self._held = False

    def _document(self) -> dict[str, Any]:
        now = time.time()
        return {
            "owner_token": self.owner_token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": now,
            "heartbeat_at": now,
            "expires_at": now + self.ttl_seconds,
        }

    def _create_exclusive(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(self._document(), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self._held = True
        return True

    def acquire(self) -> None:
        if self._held:
            raise LeaseError("lease is already held by this object")
        if self._create_exclusive():
            return
        try:
            observed = json.loads(self.path.read_text(encoding="utf-8"))
            expired = float(observed.get("expires_at", float("inf"))) <= time.time()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            expired = False
        if not expired:
            raise LeaseError("run lease is held or its state is unreadable")

        # Serialize stale recovery within this mounted filesystem. On Drive this
        # remains best effort and must not be described as a distributed mutex.
        recovery = self.path.with_name(self.path.name + ".recovering")
        try:
            guard = os.open(recovery, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise LeaseError("another process is recovering the stale lease") from exc
        os.close(guard)
        try:
            try:
                latest = json.loads(self.path.read_text(encoding="utf-8"))
                if float(latest.get("expires_at", float("inf"))) > time.time():
                    raise LeaseError("lease was renewed during stale recovery")
                stale = self.path.with_name(self.path.name + f".stale-{uuid.uuid4().hex}")
                os.replace(self.path, stale)
            except FileNotFoundError:
                pass
            if not self._create_exclusive():
                raise LeaseError("could not acquire lease after stale recovery")
        finally:
            recovery.unlink(missing_ok=True)

    def heartbeat(self) -> None:
        if not self._held:
            raise LeaseError("cannot renew a lease not held by this owner")
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._held = False
            raise LeaseError("lease disappeared before heartbeat") from exc
        if current.get("owner_token") != self.owner_token:
            self._held = False
            raise LeaseError("lease owner token changed before heartbeat")
        temp = self.path.with_name(self.path.name + f".{self.owner_token}.tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(self._document(), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.path)

    def release(self) -> None:
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._held = False
            raise LeaseError("lease is missing or unreadable") from exc
        if current.get("owner_token") != self.owner_token:
            self._held = False
            raise LeaseError("owner token does not match current lease")
        self.path.unlink()
        self._held = False
