import time
from dataclasses import dataclass, field


class AnalysisBudgetExceeded(RuntimeError):
    def __init__(self, stage: str):
        super().__init__(f"Analysis time budget exhausted during {stage}.")
        self.stage = stage


@dataclass
class AnalysisBudget:
    mode: str
    target_seconds: float
    soft_limit_seconds: float
    hard_limit_seconds: float
    finalization_reserve_seconds: float
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    def remaining(self) -> float:
        return max(0.0, self.hard_limit_seconds - self.elapsed())

    def remaining_soft(self) -> float:
        return max(0.0, self.soft_limit_seconds - self.elapsed())

    def expired(self, reserve_seconds: float = 0.0) -> bool:
        return self.remaining() <= reserve_seconds

    def can_start(
        self,
        expected_seconds: float = 0.0,
        reserve_seconds: float | None = None,
    ) -> bool:
        reserve = (
            self.finalization_reserve_seconds
            if reserve_seconds is None
            else reserve_seconds
        )
        return self.remaining() > expected_seconds + reserve

    def timeout_for(
        self,
        max_seconds: float,
        reserve_seconds: float | None = None,
        minimum_seconds: float = 0.25,
    ) -> float:
        reserve = (
            self.finalization_reserve_seconds
            if reserve_seconds is None
            else reserve_seconds
        )
        available = self.remaining() - reserve
        if available <= 0:
            raise AnalysisBudgetExceeded("deadline allocation")
        return max(minimum_seconds, min(max_seconds, available))

    def should_stop_optional_work(self) -> bool:
        return self.elapsed() >= self.soft_limit_seconds
