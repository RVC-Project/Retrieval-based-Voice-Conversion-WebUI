import logging
import os
import signal
import subprocess
import time


logger = logging.getLogger(__name__)


def kill_process_tree(process, process_name="", task_logger=None):
    """Terminate a Popen process and every child process it created."""
    if process is None:
        return False
    try:
        if process.poll() is not None:
            return False
    except (OSError, ProcessLookupError):
        return False

    pid = process.pid
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/t", "/f", "/pid", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
    else:
        try:
            process_group = os.getpgid(pid)
        except (OSError, ProcessLookupError):
            process_group = None
        try:
            if process_group is not None and process_group != os.getpgrp():
                os.killpg(process_group, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

        for _ in range(10):
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if process.poll() is None:
            try:
                if process_group is not None and process_group != os.getpgrp():
                    os.killpg(process_group, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    try:
        process.wait(timeout=5)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass

    log = task_logger or logger
    log.info("%s process tree terminated (pid=%s)", process_name or "Child", pid)
    return True
