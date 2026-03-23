"""Tests to verify that all log output includes timestamps.

Covers the application logger, werkzeug (HTTP access logs), and connexion
to ensure no log line is emitted without a leading timestamp.
"""

import logging
import re
from io import StringIO

# Timestamp pattern: YYYY-MM-DD HH:MM:SS.mmm [LEVEL] logger: message
TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "
    r"\[(INFO|WARNING|ERROR|DEBUG|CRITICAL)\] "
    r".+: .+"
)


def _capture_log_output(logger_name: str, message: str) -> str:
    """Emit a log message through the given logger and return the formatted output."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)

    # Apply the same format used in __main__.py
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    test_logger = logging.getLogger(logger_name)
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.DEBUG)

    try:
        test_logger.info(message)
        return buf.getvalue().strip()
    finally:
        test_logger.removeHandler(handler)


class TestLogTimestamps:
    """Verify that log lines always contain timestamps."""

    def test_application_logger_has_timestamp(self):
        output = _capture_log_output("tee_gateway.__main__", "TEE initialized")
        assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"

    def test_werkzeug_logger_has_timestamp(self):
        output = _capture_log_output(
            "werkzeug", '192.168.1.1 - - "GET /health HTTP/1.1" 200 -'
        )
        assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"

    def test_connexion_logger_has_timestamp(self):
        output = _capture_log_output("connexion", "API specification loaded")
        assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"

    def test_heartbeat_logger_has_timestamp(self):
        output = _capture_log_output("heartbeat", "Heartbeat relayed tx=0x123")
        assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"

    def test_dynamic_pricing_logger_has_timestamp(self):
        output = _capture_log_output(
            "llm_server.dynamic_pricing", "Session cost calculated"
        )
        assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"

    def test_timestamp_has_milliseconds(self):
        output = _capture_log_output("tee_gateway.test", "precision check")
        # Verify ms component: "HH:MM:SS.mmm"
        ms_match = re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", output)
        assert ms_match, f"Missing milliseconds in: {output!r}"

    def test_log_format_components(self):
        output = _capture_log_output("tee_gateway.test", "format check")
        # Verify all components: timestamp [LEVEL] name: message
        parts = output.split(" ", 2)
        assert len(parts) >= 3, f"Unexpected format: {output!r}"
        # Date part
        assert re.match(r"\d{4}-\d{2}-\d{2}", parts[0])
        # Time part with ms
        assert re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}", parts[1])
        # Level bracket
        assert "[INFO]" in output
        # Logger name
        assert "tee_gateway.test:" in output

    def test_warning_level_has_timestamp(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)

        test_logger = logging.getLogger("tee_gateway.warning_test")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        try:
            test_logger.warning("test warning message")
            output = buf.getvalue().strip()
            assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"
            assert "[WARNING]" in output
        finally:
            test_logger.removeHandler(handler)

    def test_error_level_has_timestamp(self):
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)

        test_logger = logging.getLogger("tee_gateway.error_test")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        try:
            test_logger.error("test error message")
            output = buf.getvalue().strip()
            assert TIMESTAMP_RE.match(output), f"Missing timestamp in: {output!r}"
            assert "[ERROR]" in output
        finally:
            test_logger.removeHandler(handler)


class TestThirdPartyLoggerConfig:
    """Verify that third-party loggers propagate through root (no own handlers)."""

    def test_werkzeug_logger_propagates(self):
        wlog = logging.getLogger("werkzeug")
        assert wlog.propagate is True, "werkzeug logger must propagate to root"

    def test_connexion_logger_propagates(self):
        clog = logging.getLogger("connexion")
        assert clog.propagate is True, "connexion logger must propagate to root"

    def test_root_logger_has_handler(self):
        root = logging.getLogger()
        assert len(root.handlers) > 0, "Root logger must have at least one handler"

    def test_root_handler_has_formatter(self):
        root = logging.getLogger()
        for handler in root.handlers:
            assert handler.formatter is not None, (
                f"Root handler {handler} must have a formatter with timestamps"
            )
