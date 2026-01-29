import threading
from enum import IntEnum

class Workflow(IntEnum):
    TREASURE = 5
    HQ       = 4
    HEAL     = 3
    DONATION = 2
    GENERIC  = 1

class WorkflowManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.active: Workflow | None = None

    def should_run(self):
        return self.state != FlowState.IDLE

    def can_run(self, wf: Workflow) -> bool:
        with self._lock:
            return self.active is None or self.active == wf

    def acquire(self, wf: Workflow) -> bool:
        with self._lock:
            if self.active is None or self.active <= wf:
                self.active = wf
                return True
            else:
                print(f"[WF] BLOCKED: {wf.name} → currently active: {self.active.name}")
                return False

    def release(self, wf: Workflow):
        with self._lock:
            if self.active == wf:
                self.active = None

    def force(self, wf: Workflow):
        with self._lock:
            self.active = wf

    def preempt_lower_priority(self, wf: Workflow):
        with self._lock:
            if self.active is None or self.active < wf:
                self.active = wf


WORKFLOW_MANAGER = WorkflowManager()
