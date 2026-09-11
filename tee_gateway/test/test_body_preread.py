"""The body of a paid POST is consumed, or refused, before the wrapped app can respond."""

import io
import json
import unittest

from tee_gateway.body_preread import MAX_PAID_REQUEST_BYTES, read_body_before_responding

PAID = ["/v1/ohttp", "/v1/chat/completions", "/v1/completions", "/v1/web_search"]


class _CountingStream(io.BytesIO):
    """Records how much of the body had been read when the inner app ran."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.read_calls = 0

    def read(self, n=-1):
        self.read_calls += 1
        return super().read(n)


def _environ(body: bytes, path="/v1/ohttp", method="POST", length=None, **extra):
    stream = _CountingStream(body)
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body) if length is None else length),
        "wsgi.input": stream,
        **extra,
    }
    return environ, stream


class TestReadBodyBeforeResponding(unittest.TestCase):
    def _run(self, environ):
        seen = {}
        status = []

        def inner(env, start_response):
            # An x402-style early answer: respond off the headers, never touch
            # the body. Record what the body stream looks like at this point.
            seen["input"] = env["wsgi.input"]
            seen["remaining_in_original"] = (
                len(self.stream.getvalue()) - self.stream.tell()
            )
            start_response("402 Payment Required", [])
            return [b"{}"]

        app = read_body_before_responding(inner, paths=PAID)
        body = b"".join(app(environ, lambda st, headers: status.append((st, headers))))
        seen["status"], seen["headers"] = status[0]
        seen["body"] = body
        return seen

    def test_paid_post_body_is_fully_read_before_dispatch(self):
        body = b"x" * (3 * 1024 * 1024)
        environ, self.stream = _environ(body)
        seen = self._run(environ)
        self.assertEqual(0, seen["remaining_in_original"])
        self.assertIsInstance(seen["input"], io.BytesIO)
        self.assertEqual(body, seen["input"].read())
        self.assertEqual(str(len(body)), environ["CONTENT_LENGTH"])
        self.assertEqual("402 Payment Required", seen["status"])

    def test_every_paid_route_is_covered(self):
        for path in PAID:
            environ, self.stream = _environ(b"body", path=path)
            seen = self._run(environ)
            self.assertEqual(0, seen["remaining_in_original"], path)

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

    def test_oversize_body_is_refused_before_payment_without_being_read(self):
        for path in PAID:
            environ, self.stream = _environ(
                b"", length=MAX_PAID_REQUEST_BYTES + 1, path=path
            )
            status = []
            app = read_body_before_responding(self._never_called, paths=PAID)
            body = b"".join(app(environ, lambda st, headers: status.append(st)))
            self.assertEqual(["413 Request Entity Too Large"], status, path)
            self.assertEqual({"error": "request body too large"}, json.loads(body))
            self.assertEqual(0, self.stream.read_calls)

    def test_chunked_body_is_buffered_and_given_a_length(self):
        body = b"y" * 1000
        environ, self.stream = _environ(
            body, length=0, HTTP_TRANSFER_ENCODING="chunked"
        )
        seen = self._run(environ)
        self.assertEqual(0, seen["remaining_in_original"])
        self.assertEqual("1000", environ["CONTENT_LENGTH"])
        self.assertNotIn("HTTP_TRANSFER_ENCODING", environ)

    def test_oversize_chunked_body_is_refused(self):
        environ, self.stream = _environ(
            b"z" * (MAX_PAID_REQUEST_BYTES + 10),
            length=0,
            HTTP_TRANSFER_ENCODING="chunked",
        )
        status = []
        app = read_body_before_responding(self._never_called, paths=PAID)
        list(app(environ, lambda st, headers: status.append(st)))
        self.assertEqual(["413 Request Entity Too Large"], status)

    def test_bodyless_and_malformed_length_requests_pass_through(self):
        environ, self.stream = _environ(b"", length=0)
        seen = self._run(environ)
        self.assertIs(seen["input"], self.stream)
        environ, self.stream = _environ(b"abc", length="not-a-number")
        seen = self._run(environ)
        self.assertIs(seen["input"], self.stream)

    @staticmethod
    def _never_called(environ, start_response):
        raise AssertionError("inner app must not run for a refused body")


if __name__ == "__main__":
    unittest.main()
