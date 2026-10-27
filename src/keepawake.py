"""Stop Windows from sleeping while a long job runs (the laptop crashes in Modern Standby sleep).

Uses SetThreadExecutionState, the same request video players use. It changes no settings and
ends automatically when the program exits. The screen may still turn off; the computer keeps running.

    from keepawake import keep_awake
    with keep_awake():
        long_job()
"""
import contextlib
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake():
    if sys.platform != "win32":
        yield
        return
    import ctypes
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    ok = set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not ok:
        print("warning: could not block sleep; set 'Sleep: Never' while plugged in")
    try:
        yield
    finally:
        set_state(ES_CONTINUOUS)
