"""The body of a paid POST is consumed before the wrapped app can respond."""

import io
import unittest

from tee_gateway.__main__ import MAX_PREREAD_REQUEST_BYTES, _read_body_before_responding


class _CountingStream(io.BytesIO):
    """Records how much of the body had been read when the inner app ran."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.read_calls = 0

    def read(self, n=-1):
        self.read_calls += 1
        return super().read(n)


def _environ(body: bytes, path="/v1/ohttp", method="POST", length=None):
    stream = _CountingStream(body)
    return {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body) if length is None else length),
        "wsgi.input": stream,
    }, stream


class TestReadBodyBeforeResponding(unittest.TestCase):
    def _run(self, environ):
        seen = {}

        def inner(env, start_response):
            # An x402-style early answer: respond off the headers, never touch
            # the body. Record what the body stream looks like at this point.
            seen["input"] = env["wsgi.input"]
            seen["remaining_in_original"] = (
                len(self.stream.getvalue()) - self.stream.tell()
            )
            start_response("402 Payment Required", [])
            return [b"{}"]

        app = _read_body_before_responding(
            inner, paths=["/v1/ohttp", "/v1/chat/completions"]
        )
        list(app(environ, lambda *a: None))
        return seen

    def test_paid_post_body_is_fully_read_before_dispatch(self):
        body = b"x" * (3 * 1024 * 1024)
        environ, self.stream = _environ(body)
        seen = self._run(environ)
        self.assertEqual(0, seen["remaining_in_original"])
        self.assertIsInstance(seen["input"], io.BytesIO)
        self.assertEqual(body, seen["input"].read())
        self.assertEqual(str(len(body)), environ["CONTENT_LENGTH"])

    def test_other_paths_and_methods_are_untouched(self):
        for path, method in (
            ("/health", "GET"),
            ("/v1/keys", "POST"),
            ("/v1/ohttp", "GET"),
        ):
            environ, self.stream = _environ(b"body", path=path, method=method)
            seen = self._run(environ)
            self.assertIs(seen["input"], self.stream, (path, method))
            self.assertEqual(0, self.stream.read_calls)

    def test_oversize_and_bodyless_requests_are_not_buffered(self):
        environ, self.stream = _environ(b"", length=MAX_PREREAD_REQUEST_BYTES + 1)
        seen = self._run(environ)
        self.assertIs(seen["input"], self.stream)
        environ, self.stream = _environ(b"", length=0)
        seen = self._run(environ)
        self.assertIs(seen["input"], self.stream)
        environ, self.stream = _environ(b"abc", length="not-a-number")
        seen = self._run(environ)
        self.assertIs(seen["input"], self.stream)


if __name__ == "__main__":
    unittest.main()
