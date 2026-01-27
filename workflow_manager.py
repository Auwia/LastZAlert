import threading
from enum import IntEnum

class Workflow(IntEnum):
    TREASURE = 1
    HQ       = 2
    HEAL     = 3
    GENERIC  = 4

class WorkflowManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.current = None
        self.active: Workflow | None = None

    def can_run(self, wf: Workflow) -> bool:
        with self._lock:
            if self.active is None:
                return True
            return wf <= self.active  # priorità più alta vince

    def acquire(self, wf: Workflow) -> bool:
        with self._lock:
            if self.active is None or wf < self.active:
                self.active = wf
                return True
            return False

    def release(self, wf: Workflow):
        with self._lock:
            if self.active == wf:
                self.active = None

    def is_idle_or(self, wf: Workflow) -> bool:
        with self._lock:
            return self.current is None or self.current == wf

    def preempt_lower_priority(self, wf: Workflow):
        with self._lock:
            if self.current != wf:
                self.current = wf

    def force(self, wf: Workflow):
        """
        Forza l'attivazione di un workflow
        scacciando qualsiasi altro workflow attivo
        """
        with self._lock:
            self.current = wf

# istanza globale
WORKFLOW_MANAGER = WorkflowManager()

