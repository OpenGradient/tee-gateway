"""Read a paid POST's body before anything downstream can answer.

The x402 payment middleware decides from the request *headers* whether to
answer ``402 Payment Required``, before a byte of the body has been read. On a
multi-megabyte body (an image attachment) that 402 leaves the enclave while
nitriding's Go reverse proxy is still receiving the upload from the relay.
Go's HTTP server will not drain more than 256 KiB of unread body after a
response: it sends its FIN, waits 500 ms and closes, which is a TCP reset
while the relay is still uploading. gvproxy (the host-side vsock forwarder)
relays the abort, so the relay sees a connection reset and no response at all
(httpx ``ReadError`` with an empty message), or nitriding sees the reset first
and answers a bare 502. A request that only needed a payment challenge fails,
and it fails exactly on big bodies, which is what "flaky 502s on image chats"
looked like from the app.

:func:`read_body_before_responding` wraps the WSGI stack *outside* the
payment middleware and buffers the body of a POST to a paid route before
dispatch, so every response, the 402 included, is written after the upload has
been consumed end to end and the connection is left clean for the paid retry.
The handlers already buffer the whole body (``get_data``), so this changes
where the bytes are read, not how many are held.

A body larger than :data:`MAX_PAID_REQUEST_BYTES` is refused with **413**
right here, before payment. Nothing in the enclave accepts such a body (the
OHTTP handler re-checks the same cap after reading), so buffering it would
only cost memory; and answering 402 for it would recreate the reset above. The
413 is written without reading the body, so an oversize upload may still see
a reset instead of the status; that is the one case where this is acceptable.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable
from typing import Any

# The largest request body any paid route accepts. The OHTTP handler enforces
# the same number on the decrypted side; keep them one constant.
MAX_PAID_REQUEST_BYTES = 20 * 1024 * 1024

WSGIApp = Callable[[dict[str, Any], Callable[..., Any]], Iterable[bytes]]


def _too_large(start_response: Callable[..., Any]) -> Iterable[bytes]:
    body = json.dumps({"error": "request body too large"}).encode()
    start_response(
        "413 Request Entity Too Large",
        [("Content-Type", "application/json"), ("Content-Length", str(len(body)))],
    )
    return [body]


def read_body_before_responding(wsgi_app: WSGIApp, paths: Iterable[str]) -> WSGIApp:
    """WSGI wrapper: buffer the body of a POST to ``paths`` before dispatch."""
    paid_paths = frozenset(paths)

    def wrapper(
        environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> Iterable[bytes]:
        if (
            environ.get("REQUEST_METHOD") != "POST"
            or environ.get("PATH_INFO") not in paid_paths
        ):
            return wsgi_app(environ, start_response)

        try:
            declared = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            declared = 0
        chunked = "chunked" in environ.get("HTTP_TRANSFER_ENCODING", "").lower()

        if declared > MAX_PAID_REQUEST_BYTES:
            return _too_large(start_response)
        if declared > 0:
            body = environ["wsgi.input"].read(declared)
        elif chunked:
            # No declared length: read up to the cap plus one byte so an
            # oversize chunked upload is refused instead of buffered.
            body = environ["wsgi.input"].read(MAX_PAID_REQUEST_BYTES + 1)
            if len(body) > MAX_PAID_REQUEST_BYTES:
                return _too_large(start_response)
        else:
            return wsgi_app(environ, start_response)

        environ["wsgi.input"] = io.BytesIO(body)
        environ["CONTENT_LENGTH"] = str(len(body))
        environ.pop("HTTP_TRANSFER_ENCODING", None)
        return wsgi_app(environ, start_response)

    return wrapper
