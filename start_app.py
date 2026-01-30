#!/usr/bin/env python
"""Start the COT dashboard in the background."""
import os
import signal
import subprocess
import sys
from pathlib import Path

PID_FILE = Path(__file__).resolve().parent / ".app.pid"
APP_SCRIPT = Path(__file__).resolve().parent / "apps" / "app.py"


def main():
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"App is already running (PID {pid}). Stop it first with stop_app.py")
            sys.exit(1)
        except OSError:
            PID_FILE.unlink()

    proc = subprocess.Popen(
        [sys.executable, str(APP_SCRIPT)],
        stdout=open("app.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    PID_FILE.write_text(str(proc.pid))
    print(f"Dashboard started (PID {proc.pid})")
    print("  URL:  http://127.0.0.1:8050")
    print("  Log:  app.log")
    print("  Stop: python stop_app.py")


if __name__ == "__main__":
    main()
