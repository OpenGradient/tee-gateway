"""gunicorn settings for serving the gateway inside the enclave.

Until this file existed the enclave ran ``application.run()`` — Werkzeug's
development server: one unbounded thread per connection, a 128-entry listen
backlog, ``Connection: close`` on every response, no request logging worth the
name, and a banner in its own docs saying not to use it in production. Load
showed up only as thread growth and, past the backlog, as bare 502s from
nitriding's reverse proxy.

Everything here follows from one constraint: **the gateway is one process**.
The TEE signing key is generated at import and its hash registered with
nitriding once; provider API keys are injected once over ``POST /v1/keys``
into that process; the x402 session store is an in-memory dict. None of that
survives a fork or a restart, so:

* ``workers = 1`` with the ``gthread`` worker — concurrency comes from
  threads in the one process, as before, but bounded and with a real backlog.
* No ``preload_app``: the app (and the key material) is created in the worker,
  exactly as when it ran directly.
* A worker exit is fatal (``child_exit`` below). gunicorn would otherwise
  fork a fresh worker that generates a *new* signing key — while the registry
  and nitriding still hold the old one — and has no provider keys, and would
  answer every request wrongly until someone noticed. Dying loudly, as the
  bare process did, is the safer behaviour; the host's supervision decides
  what happens next.

Connections to this server come only from nitriding's Go reverse proxy on
loopback. Go's transport pools idle connections indefinitely (nitriding sets
no IdleConnTimeout) and does not retry a POST it has already written, so the
server must never be the side that closes an idle connection first; hence a
keep-alive far longer than any realistic idle gap.
"""

from __future__ import annotations

import os

bind = (
    f"{os.getenv('API_SERVER_HOST', '0.0.0.0')}:{os.getenv('API_SERVER_PORT', '8000')}"
)

# One process; see the module docstring for why this must stay 1.
workers = 1
worker_class = "gthread"
# Concurrent requests the process serves. Each in-flight chat stream holds a
# thread for its whole duration (blocking provider I/O), so this is the cap on
# simultaneous conversations per enclave. Requests beyond it wait in the
# backlog instead of spawning threads without limit.
threads = int(os.getenv("GUNICORN_THREADS", "64"))
worker_connections = 1000
backlog = 2048

# Never let this server close an idle keep-alive connection before the Go
# proxy does (see module docstring); an hour is "never" for loopback traffic.
keepalive = 3600

# The gthread worker heartbeats from its main loop, so a long-running request
# cannot trip this — but a GIL-starved main loop under heavy load could, and
# with child_exit fatal that would take the enclave down. Werkzeug had no such
# watchdog; keep none.
timeout = 0
# On SIGTERM, let in-flight streams finish (chat-api waits up to 210 s).
graceful_timeout = 240

# No recycling: a worker restart is a key rotation (see module docstring).
max_requests = 0

# stdout/stderr are the enclave console, like the app's own logger.
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'


def child_exit(server, worker):  # noqa: ANN001 - gunicorn hook signature
    """Stop the whole server when the single worker exits.

    Runs in the arbiter. A replacement worker would carry a new TEE signing
    key and no provider keys (both are process state), so serving on is worse
    than stopping: attestation and signature verification would fail for
    every client until the enclave is restarted anyway.
    """
    server.log.critical(
        "gateway worker %s exited; TEE key material, injected provider keys and "
        "x402 sessions lived in that process — halting instead of serving with "
        "regenerated keys",
        worker.pid,
    )
    server.halt(reason="gateway worker exited", exit_status=1)
