#!/usr/bin/env python
"""Stop the COT dashboard background process."""
import os
import signal
import sys
from pathlib import Path

PID_FILE = Path(__file__).resolve().parent / ".app.pid"


def main():
    if not PID_FILE.exists():
        print("No running app found (.app.pid missing).")
        sys.exit(0)

    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Dashboard stopped (PID {pid}).")
    except ProcessLookupError:
        print(f"Process {pid} was not running.")
    finally:
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
