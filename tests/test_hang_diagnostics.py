"""The engine side of "cancel a stuck run and say where it was stuck":
step start times in the status file, and the periodic stack dump."""

import subprocess
import sys
import time

from soundboard_lab.cli import StepTracker

STUCK_REPAIR_SCRIPT = """
import sys, time
import soundboard_lab.cli as cli

cli.STACK_DUMP_INTERVAL_S = 1
cli.repair_stl = lambda *a, **k: time.sleep(60)  # stand-in for a hung native call
sys.exit(cli.main(sys.argv[1:]))
"""


def test_step_tracker_records_start_time(tmp_path):
    status_path = tmp_path / "status.json"
    tracker = StepTracker(status_path, [("repair", "Repair mesh"), ("mesh", "Build shell mesh")])
    before = time.time()
    tracker.start("repair")

    import json

    steps = {s["id"]: s for s in json.loads(status_path.read_text())["steps"]}
    assert before <= steps["repair"]["started_at"] <= time.time()
    assert "started_at" not in steps["mesh"]


def test_stack_log_shows_where_a_stuck_run_is(tmp_path):
    stack_log = tmp_path / "stack.log"
    process = subprocess.Popen(
        [
            sys.executable, "-c", STUCK_REPAIR_SCRIPT,
            "--stl", str(tmp_path / "any.stl"),
            "--ccx-exe", "ccx",
            "--output-json", str(tmp_path / "result.json"),
            "--mesh-type", "shell",
            "--status-json", str(tmp_path / "status.json"),
            "--stack-log", str(stack_log),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline and "Timeout" not in (stack_log.read_text() if stack_log.exists() else ""):
            time.sleep(0.25)
        dump = stack_log.read_text()
    finally:
        process.kill()
        process.wait()

    assert "Timeout" in dump
    assert "<lambda>" in dump and "run_shell_analysis" in dump
