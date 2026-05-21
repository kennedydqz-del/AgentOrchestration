import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.agent.runtime import AgentRuntime, FailureReason, RuntimeState


class TestAgentRuntime:
    def setup_method(self):
        self.runtime = AgentRuntime()

    # ---- State machine guard ----

    def test_valid_start_to_running(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            assert self.runtime.start("agent-1", ["echo", "hello"])
            assert self.runtime.get_state("agent-1") == RuntimeState.RUNNING

    def test_state_transition_guard_rejects_invalid(self):
        runtime = AgentRuntime()
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            runtime._set_state("agent-1", RuntimeState.CRASHED)

    def test_stopped_cannot_transition_to_running(self):
        runtime = AgentRuntime()
        runtime._set_state("agent-1", RuntimeState.STOPPED)
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            runtime._set_state("agent-1", RuntimeState.RUNNING)

    def test_running_cannot_transition_to_starting(self):
        runtime = AgentRuntime()
        runtime._set_state("agent-1", RuntimeState.STARTING)
        runtime._set_state("agent-1", RuntimeState.RUNNING)
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            runtime._set_state("agent-1", RuntimeState.STARTING)

    def test_crashed_can_only_transition_to_stopped(self):
        runtime = AgentRuntime()
        runtime._set_state("agent-1", RuntimeState.STARTING)
        runtime._set_state("agent-1", RuntimeState.CRASHED)
        # valid: CRASHED -> STOPPED
        runtime._set_state("agent-1", RuntimeState.STOPPED)
        # Use a new agent to test CRASHED -> RUNNING is invalid
        runtime._set_state("agent-2", RuntimeState.STARTING)
        runtime._set_state("agent-2", RuntimeState.CRASHED)
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            runtime._set_state("agent-2", RuntimeState.RUNNING)

    # ---- Failure reason recording ----

    def test_failure_reason_on_start_exception(self):
        with patch("subprocess.Popen", side_effect=OSError("No such file")):
            result = self.runtime.start("agent-2", ["nonexistent"])
            assert not result
            assert self.runtime.get_state("agent-2") == RuntimeState.CRASHED
            failure = self.runtime.get_failure_reason("agent-2")
            assert failure is not None
            assert failure.exit_code == -1
            assert "No such file" in failure.stderr_tail

    def test_failure_reason_on_nonzero_exit(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12346
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-3", ["python", "-c", "exit(1)"])
            assert self.runtime.get_state("agent-3") == RuntimeState.RUNNING

            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"Error: something went wrong")

            self.runtime.stop("agent-3")
            assert self.runtime.get_state("agent-3") == RuntimeState.STOPPED
            failure = self.runtime.get_failure_reason("agent-3")
            assert failure is not None
            assert failure.exit_code == 1
            assert "Error: something went wrong" in failure.stderr_tail

    def test_failure_reason_detected_by_get_state(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12347
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-4", ["python", "-c", "exit(2)"])
            assert self.runtime.is_running("agent-4")

            # Simulate crash: poll returns non-None non-zero
            mock_proc.poll.return_value = 2
            mock_proc.returncode = 2
            mock_proc.communicate.return_value = (b"", b"Fatal error")

            state = self.runtime.get_state("agent-4")
            assert state == RuntimeState.CRASHED
            failure = self.runtime.get_failure_reason("agent-4")
            assert failure is not None
            assert failure.exit_code == 2
            assert "Fatal error" in failure.stderr_tail

    def test_no_failure_reason_when_not_crashed(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12348
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-5", ["echo", "hello"])
            assert self.runtime.get_failure_reason("agent-5") is None

    def test_failure_reasons_are_persisted(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12349
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-a", ["echo", "a"])

            # Process exits mid-way
            mock_proc.poll.side_effect = [None, 1]
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"err a")

            self.runtime.stop("agent-a")

            # Failure reason should persist across repeated reads
            f1 = self.runtime.get_failure_reason("agent-a")
            f2 = self.runtime.get_failure_reason("agent-a")
            assert f1 == f2
            assert f1.exit_code == 1

    def test_get_all_failure_reasons(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12350
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-fail-1", ["echo", "x"])
            mock_proc.poll.side_effect = [None, 1]
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"err1")
            self.runtime.stop("agent-fail-1")

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12351
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-fail-2", ["echo", "y"])
            mock_proc.poll.side_effect = [None, 2]
            mock_proc.returncode = 2
            mock_proc.communicate.return_value = (b"", b"err2")
            self.runtime.stop("agent-fail-2")

        all_failures = self.runtime.get_failure_reasons()
        assert "agent-fail-1" in all_failures
        assert "agent-fail-2" in all_failures
        assert all_failures["agent-fail-1"].exit_code == 1
        assert all_failures["agent-fail-2"].exit_code == 2

    # ---- Stop when process already exited ----

    def test_stop_records_failure_when_already_exited(self):
        """When process has already exited before stop(), failure is recorded."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12352
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-6", ["python", "-c", "exit(4)"])
            assert self.runtime.is_running("agent-6")

            # Process already exited with non-zero
            mock_proc.poll.return_value = 4
            mock_proc.returncode = 4
            mock_proc.communicate.return_value = (
                b"",
                b"segfault at 0xdeadbeef",
            )

            result = self.runtime.stop("agent-6")
            assert result  # stop found process already terminated
            failure = self.runtime.get_failure_reason("agent-6")
            assert failure is not None
            assert failure.exit_code == 4
            assert "segfault" in failure.stderr_tail

    # ---- Durability of terminal state ----

    def test_crashed_state_persists_after_stop(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12353
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-7", ["python", "-c", "exit(3)"])
            mock_proc.poll.side_effect = [None, 3]
            mock_proc.returncode = 3
            mock_proc.communicate.return_value = (b"", b"crash info")

            self.runtime.stop("agent-7")
            failure = self.runtime.get_failure_reason("agent-7")
            assert failure is not None
            assert failure.exit_code == 3
            assert "crash info" in failure.stderr_tail

    def test_default_state_is_stopped(self):
        assert self.runtime.get_state("nonexistent") == RuntimeState.STOPPED

    # ---- Stop with timeout ----

    def test_stop_kills_after_timeout(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12354
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-8", ["sleep", "100"])
            assert self.runtime.get_state("agent-8") == RuntimeState.RUNNING

            # First wait: timeout, second wait: succeeds (after kill)
            mock_proc.wait.side_effect = [
                subprocess.TimeoutExpired("sleep", 0.1),
                None,
            ]
            mock_proc.returncode = -9  # SIGKILL
            mock_proc.communicate.return_value = (b"", b"")

            result = self.runtime.stop("agent-8", timeout=0)
            assert result
            assert self.runtime.get_state("agent-8") == RuntimeState.STOPPED
            mock_proc.kill.assert_called_once()

    # ---- Concurrency: no orphaned state ----

    def test_duplicate_start_returns_false(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12355
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            assert self.runtime.start("agent-9", ["echo", "hello"])
            assert not self.runtime.start("agent-9", ["echo", "duplicate"])

    def test_cancellation_leaves_durable_terminal_state(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12356
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-cancel", ["long-running-task"])
            assert self.runtime.is_running("agent-cancel")

            # Signal cancellation: SIGTERM sends 143 (128+15)
            mock_proc.returncode = 143
            mock_proc.communicate.return_value = (
                b"",
                b"terminated by signal",
            )

            self.runtime.stop("agent-cancel")
            state = self.runtime.get_state("agent-cancel")
            assert state == RuntimeState.STOPPED

            failure = self.runtime.get_failure_reason("agent-cancel")
            assert failure is not None
            assert failure.exit_code == 143

    # ---- Zero exit code is not a failure ----

    def test_zero_exit_does_not_record_failure(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12357
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            self.runtime.start("agent-ok", ["echo", "success"])
            mock_proc.returncode = 0

            self.runtime.stop("agent-ok")
            assert self.runtime.get_state("agent-ok") == RuntimeState.STOPPED
            assert self.runtime.get_failure_reason("agent-ok") is None
