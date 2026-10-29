"""Bound native SDK destruction after commanding has already stopped."""
import threading

from loguru import logger


def close_sdk_channel(channel, timeout_s=0.5):
    errors = []

    def close():
        try:
            channel.Close()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=close, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        logger.warning("SDK channel destruction timed out after {}s; commanding has stopped", timeout_s)
        return False
    if errors:
        logger.warning("SDK channel destruction failed: {}", errors[0])
        return False
    return True


def exit_after_sdk_timeout(timed_out):
    if timed_out:
        # Native reader deletion can wait for a callback that needs the GIL.
        # Motor fallback and recordings must be finished BEFORE this call.
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
