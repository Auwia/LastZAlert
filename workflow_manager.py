import threading
from enum import IntEnum

class Workflow(IntEnum):
    TREASURE = 10
    HQ       = 9
    HEAL     = 8
    DONATION = 7
    MINISTRY = 6
    FORZIERE = 5
    GENERIC  = 4
    RESEARCH = 3
    RALLY    = 2
    HERO     = 1
    BOUNTY   = 0

class WorkflowManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.active: Workflow | None = None

    def has_active(self) -> bool:
        return self.active is not None

    def is_idle(self) -> bool:
        return self.active is None

    def current(self):
        return self.active

    def is_active(self, wf: Workflow) -> bool:
        return self.active == wf

    def should_run(self):
        return self.state != FlowState.IDLE

    def can_run(self, wf: Workflow) -> bool:
        with self._lock:
            return self.active is None or self.active == wf

    def acquire(self, wf: Workflow) -> bool:
        with self._lock:
            # libero -> acquisisce
            if self.active is None:
                self.active = wf
                return True
    
            # stesso workflow -> può continuare
            if self.active == wf:
                return True
    
            # qualsiasi altro workflow è già attivo -> aspetta
            return False

    def release(self, wf: Workflow):
        #print(f"[WF] release {Workflow}")
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
