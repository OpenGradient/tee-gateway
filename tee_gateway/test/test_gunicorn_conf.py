"""The single worker's death halts gunicorn; a normal shutdown does not trip it."""

import logging
import unittest
from unittest import mock

from tee_gateway import gunicorn_conf


class _Server:
    def __init__(self, listeners):
        self.LISTENERS = listeners
        self.log = logging.getLogger("test.gunicorn")
        self.halt = mock.Mock()


class TestChildExit(unittest.TestCase):
    def test_unsolicited_worker_exit_halts_with_status_1(self):
        server = _Server(listeners=["a socket"])
        gunicorn_conf.child_exit(server, mock.Mock(pid=4242))
        server.halt.assert_called_once_with(
            reason="gateway worker exited", exit_status=1
        )

    def test_worker_exit_during_shutdown_is_left_to_the_arbiter(self):
        # Arbiter.stop() empties LISTENERS before signalling the worker.
        server = _Server(listeners=[])
        gunicorn_conf.child_exit(server, mock.Mock(pid=4242))
        server.halt.assert_not_called()

    def test_one_gthread_worker(self):
        self.assertEqual(1, gunicorn_conf.workers)
        self.assertEqual("gthread", gunicorn_conf.worker_class)
        self.assertEqual(0, gunicorn_conf.max_requests)
        self.assertFalse(getattr(gunicorn_conf, "preload_app", False))


if __name__ == "__main__":
    unittest.main()
