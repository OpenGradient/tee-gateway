"""Error-only, content-free diagnostics over vsock (production enclaves).

Host: python3 tee_gateway/error_logging.py --collect logs/tee-errors.log
The collector daemonizes after binding; repeated starts reuse its file lock.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import socket
import threading
import traceback

_PORT = 1025
_MAX_EVENT = 8192


def _diagnostic(record: logging.LogRecord) -> bytes:
    # Never format the message, args, source lines, or locals. SDK exception
    # messages and even logging templates can contain prompts/credentials.
    event = {
        "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name[:128],
        "location": f"{os.path.basename(record.pathname)}:{record.lineno}:{record.funcName}",
        "exceptions": [],
    }
    exc = record.exc_info[1] if record.exc_info else None
    seen = set()
    while exc is not None and id(exc) not in seen and len(seen) < 4:
        seen.add(id(exc))
        detail = {
            "type": type(exc).__name__,
            "frames": [
                f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
                for frame in traceback.extract_tb(exc.__traceback__)[-6:]
            ],
        }
        status = getattr(exc, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            detail["status"] = status
        event["exceptions"].append(detail)
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return json.dumps(event, separators=(",", ":")).encode() + b"\n"


class EnclaveErrorHandler(logging.Handler):
    """Bounded best-effort export; collector failures never block inference."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self._events: queue.Queue[bytes] = queue.Queue(maxsize=128)
        threading.Thread(target=self._send, daemon=True, name="error-export").start()

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR:
            return
        try:
            payload = _diagnostic(record)
            if len(payload) <= _MAX_EVENT:
                self._events.put_nowait(payload)
        except Exception:
            # No recursive logging or stderr fallback containing the record.
            pass

    def _send(self) -> None:
        while True:
            payload = self._events.get()
            try:
                with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    sock.connect((socket.VMADDR_CID_HOST, _PORT))
                    sock.sendall(payload)
            except OSError:
                pass
            finally:
                self._events.task_done()


def install_error_export(logger: logging.Logger | None = None) -> None:
    # No vsock on development Macs. Called again after Gunicorn forks so no
    # worker relies on a sender thread that existed only in the parent.
    if not hasattr(socket, "AF_VSOCK"):
        return
    target = logger if logger is not None else logging.getLogger()
    for handler in list(target.handlers):
        if isinstance(handler, EnclaveErrorHandler):
            target.removeHandler(handler)
    target.addHandler(EnclaveErrorHandler())


def collect(log_path: str) -> None:
    import fcntl  # Linux host only; keep the enclave/development import portable.

    os.umask(0o077)
    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(path) + ".lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Error collector already running: {path}")
        return

    listener = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    listener.bind((socket.VMADDR_CID_ANY, _PORT))
    listener.listen(32)
    handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter("%(message)s"))
    if os.fork():
        print(f"Error collector started: {path}")
        return

    os.setsid()
    # Keep the lock fd for the daemon lifetime. Detach terminal output so the
    # restart command exits normally even when invoked through SSH.
    with open(os.devnull, "r+b", buffering=0) as null:
        for fd in (0, 1, 2):
            os.dup2(null.fileno(), fd)
    try:
        while True:
            conn, peer = listener.accept()
            with conn:
                if peer[0] != 4:  # scripts/run-enclave.sh pins enclave CID 4.
                    continue
                conn.settimeout(0.5)
                try:
                    data = bytearray()
                    while b"\n" not in data and len(data) <= _MAX_EVENT:
                        chunk = conn.recv(min(4096, _MAX_EVENT + 1 - len(data)))
                        if not chunk:
                            break
                        data.extend(chunk)
                    if len(data) > _MAX_EVENT:
                        continue
                    event = json.loads(data)
                    if not isinstance(event, dict) or event.get("level") not in (
                        "ERROR",
                        "CRITICAL",
                    ):
                        continue
                    # Serialize again: no literal newlines/control sequences
                    # from the wire can forge additional host log entries.
                    line = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
                    handler.handle(
                        logging.LogRecord(
                            "tee-errors", logging.ERROR, "", 0, line, (), None
                        )
                    )
                except (OSError, ValueError):
                    continue
    finally:
        listener.close()
        handler.close()
        lock.close()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", required=True, metavar="LOG_FILE")
    collect(parser.parse_args().collect)
