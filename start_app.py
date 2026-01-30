#!/usr/bin/env python
"""Start the COT dashboard in the background."""
import os
import subprocess
import sys
import time
from pathlib import Path

PID_FILE = Path(__file__).resolve().parent / ".app.pid"
APP_SCRIPT = Path(__file__).resolve().parent / "apps" / "app.py"
LOG_FILE = Path(__file__).resolve().parent / "app.log"


def main():
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"App is already running (PID {pid}). Stop it first with stop_app.py")
            sys.exit(1)
        except OSError:
            PID_FILE.unlink()

    log_fh = open(LOG_FILE, "w")
    proc = subprocess.Popen(
        [sys.executable, str(APP_SCRIPT)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Wait a moment to see if the process survives startup
    time.sleep(3)
    poll = proc.poll()
    if poll is not None:
        log_fh.close()
        print(f"App failed to start (exit code {poll}). Check app.log:")
        print(LOG_FILE.read_text())
        PID_FILE.unlink(missing_ok=True)
        sys.exit(1)

    PID_FILE.write_text(str(proc.pid))
    print(f"Dashboard started (PID {proc.pid})")
    print("  URL:  http://0.0.0.0:8050")
    print("  Log:  app.log")
    print("  Stop: python stop_app.py")


if __name__ == "__main__":
    main()
