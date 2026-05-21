"""Agent Runtime — Manages agent process lifecycle."""

import os
import signal
import subprocess
import logging
from enum import Enum
from typing import Dict, Optional, NamedTuple

logger = logging.getLogger(__name__)


class RuntimeState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    CRASHED = "crashed"


class FailureReason(NamedTuple):
    exit_code: int
    stderr_tail: str


# Valid state transitions enforced before every state assignment
_VALID_TRANSITIONS: Dict[Optional[RuntimeState], set[RuntimeState]] = {
    None: {RuntimeState.STOPPED, RuntimeState.STARTING},
    RuntimeState.STARTING: {RuntimeState.RUNNING, RuntimeState.CRASHED},
    RuntimeState.RUNNING: {RuntimeState.STOPPING, RuntimeState.CRASHED},
    RuntimeState.STOPPING: {RuntimeState.STOPPED, RuntimeState.CRASHED},
    RuntimeState.CRASHED: {RuntimeState.STOPPED},
    RuntimeState.STOPPED: set(),
}


class AgentRuntime:
    def __init__(self):
        self._processes: Dict[str, subprocess.Popen] = {}
        self._states: Dict[str, RuntimeState] = {}
        self._failures: Dict[str, FailureReason] = {}

    def _set_state(self, agent_id: str, target: RuntimeState) -> None:
        current = self._states.get(agent_id)
        allowed = _VALID_TRANSITIONS.get(current, {target})
        if target not in allowed:
            raise RuntimeError(
                f"Invalid state transition for {agent_id}: "
                f"{current} -> {target}"
            )
        self._states[agent_id] = target

    def _record_failure(
        self, agent_id: str, exit_code: int, stderr_tail: str
    ) -> None:
        self._failures[agent_id] = FailureReason(
            exit_code=exit_code,
            stderr_tail=stderr_tail,
        )
        logger.warning(
            f"Agent {agent_id} failure recorded (exit code: {exit_code})"
        )

    def start(
        self, agent_id: str, command: list, env: Optional[Dict] = None
    ) -> bool:
        if (
            agent_id in self._processes
            and self._processes[agent_id].poll() is None
        ):
            logger.warning(f"Agent {agent_id} is already running")
            return False

        self._set_state(agent_id, RuntimeState.STARTING)
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        process_env["AO_AGENT_ID"] = agent_id

        try:
            proc = subprocess.Popen(
                command,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._processes[agent_id] = proc
            self._set_state(agent_id, RuntimeState.RUNNING)
            logger.info(f"Agent {agent_id} started (PID: {proc.pid})")
            return True
        except Exception as e:
            self._record_failure(agent_id, -1, str(e))
            self._set_state(agent_id, RuntimeState.CRASHED)
            logger.error(f"Failed to start agent {agent_id}: {e}")
            return False

    def stop(self, agent_id: str, timeout: int = 10) -> bool:
        proc = self._processes.get(agent_id)
        if not proc:
            return False

        self._set_state(agent_id, RuntimeState.STOPPING)

        exit_code = proc.poll()
        if exit_code is not None:
            return self._handle_fatal_exit(agent_id, exit_code, proc)

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        exit_code = proc.returncode
        if exit_code != 0:
            _, stderr_data = proc.communicate()
            self._record_failure(
                agent_id,
                exit_code,
                stderr_data.decode(errors="replace")[-1024:]
                if stderr_data
                else "",
            )

        self._set_state(agent_id, RuntimeState.STOPPED)
        logger.info(
            f"Agent {agent_id} stopped (exit code: {exit_code})"
        )
        return True

    def _handle_fatal_exit(
        self,
        agent_id: str,
        exit_code: int,
        proc: subprocess.Popen,
    ) -> bool:
        if exit_code != 0:
            _, stderr_data = proc.communicate()
            self._record_failure(
                agent_id,
                exit_code,
                stderr_data.decode(errors="replace")[-1024:]
                if stderr_data
                else "",
            )
        self._set_state(
            agent_id,
            RuntimeState.CRASHED if exit_code != 0 else RuntimeState.STOPPED,
        )
        logger.info(
            f"Agent {agent_id} already exited with code {exit_code}"
        )
        return True

    def get_state(self, agent_id: str) -> RuntimeState:
        proc = self._processes.get(agent_id)
        current = self._states.get(agent_id, RuntimeState.STOPPED)
        if proc and proc.poll() is not None:
            if current in (RuntimeState.RUNNING, RuntimeState.STOPPING):
                exit_code = proc.returncode
                if exit_code != 0:
                    _, stderr_data = proc.communicate()
                    self._record_failure(
                        agent_id,
                        exit_code,
                        stderr_data.decode(errors="replace")[-1024:]
                        if stderr_data
                        else "",
                    )
                self._set_state(agent_id, RuntimeState.CRASHED)
        return self._states.get(agent_id, RuntimeState.STOPPED)

    def is_running(self, agent_id: str) -> bool:
        proc = self._processes.get(agent_id)
        return proc is not None and proc.poll() is None

    def get_failure_reason(self, agent_id: str) -> Optional[FailureReason]:
        return self._failures.get(agent_id)

    def get_failure_reasons(self) -> Dict[str, FailureReason]:
        return dict(self._failures)
