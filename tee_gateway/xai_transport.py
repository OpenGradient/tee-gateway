"""Replace failed xAI connection pools without interrupting other streams."""

import logging
from threading import Lock
from collections.abc import Iterator

import httpx

logger = logging.getLogger(__name__)

# Pool exhaustion is not a broken connection. Neither are provider HTTP errors
# or local request validation errors, so those must not trigger pool replacement.
_CONNECTION_ERRORS = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
)


class XaiRecoveryTransport(httpx.BaseTransport):
    """Keep the client identity stable for cached/bound ChatXAI models.

    OpenAI's SDK owns the bounded retry before response headers arrive. We
    only retire the failed pool so that retry uses fresh connections. Body
    failures also retire the pool, but never replay an already-started stream.
    Retired pools close only after all their responses have been released.
    """

    def __init__(self, *, limits: httpx.Limits) -> None:
        self._limits = limits
        self._lock = Lock()
        self._current: httpx.HTTPTransport | None = None
        self._active: dict[httpx.HTTPTransport, int] = {}
        self._closed = False

    def _acquire(self) -> httpx.HTTPTransport:
        with self._lock:
            if self._closed:
                raise RuntimeError("xAI transport is closed")
            if self._current is None:
                self._current = httpx.HTTPTransport(http2=True, limits=self._limits)
                self._active[self._current] = 0
            self._active[self._current] += 1
            return self._current

    def revive(self, failed: httpx.HTTPTransport, error: Exception) -> None:
        """Retire only the failed generation, not a concurrent replacement."""
        with self._lock:
            if self._current is not failed:
                return
            self._current = None
        # Do not log exception text: it can include request information.
        logger.warning("Retiring xAI connection pool after %s", type(error).__name__)

    def _release(self, transport: httpx.HTTPTransport) -> None:
        with self._lock:
            self._active[transport] -= 1
            should_close = (
                self._active[transport] == 0 and transport is not self._current
            )
            if should_close:
                del self._active[transport]
        if should_close:
            self._close_pool(transport)

    @staticmethod
    def _close_pool(transport: httpx.HTTPTransport) -> None:
        try:
            transport.close()
        except Exception as error:
            logger.warning("Closing retired xAI pool failed: %s", type(error).__name__)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._acquire()
        try:
            response = transport.handle_request(request)
        except BaseException as error:
            if isinstance(error, _CONNECTION_ERRORS):
                self.revive(transport, error)
            self._release(transport)
            raise
        response.stream = _RecoveryStream(response.stream, self, transport)
        return response

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._current = None
            idle = [pool for pool, count in self._active.items() if count == 0]
            for pool in idle:
                del self._active[pool]
        for pool in idle:
            self._close_pool(pool)


class _RecoveryStream(httpx.SyncByteStream):
    def __init__(
        self,
        stream: httpx.SyncByteStream,
        owner: XaiRecoveryTransport,
        transport: httpx.HTTPTransport,
    ) -> None:
        self._stream = stream
        self._owner = owner
        self._transport = transport
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self._stream
        except _CONNECTION_ERRORS as error:
            self._owner.revive(self._transport, error)
            raise
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        finally:
            self._owner._release(self._transport)
