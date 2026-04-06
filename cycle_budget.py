"""Per-cycle API budget enforcement for early-cycle Tavily and LLM usage."""

from config import MAX_LLM_CALLS_PER_CYCLE, MAX_TAVILY_CALLS_PER_CYCLE


class CycleBudgetExhausted(RuntimeError):
    """Raised when a cycle-local API budget would be exceeded."""

    def __init__(
        self,
        *,
        service: str,
        outcome: str,
        callsite: str,
        used: int,
        limit: int,
        snapshot: dict[str, int],
    ) -> None:
        self.service = str(service or "").strip()
        self.outcome = str(outcome or "").strip()
        self.callsite = str(callsite or "").strip()
        self.used = int(used)
        self.limit = int(limit)
        self.snapshot = dict(snapshot)
        super().__init__(self.summary)

    @property
    def summary(self) -> str:
        return (
            f"{self.service} cycle budget exhausted at {self.callsite} "
            f"({self.used}/{self.limit})"
        )

    def to_diagnostic(self) -> dict[str, object]:
        return {
            "service": self.service,
            "callsite": self.callsite,
            "used": self.used,
            "limit": self.limit,
            "summary": self.summary,
            "snapshot": dict(self.snapshot),
        }


class CycleBudget:
    """Track cycle-local API usage independently from daily logging."""

    def __init__(
        self,
        *,
        max_tavily_calls: int = MAX_TAVILY_CALLS_PER_CYCLE,
        max_llm_calls: int = MAX_LLM_CALLS_PER_CYCLE,
    ) -> None:
        self.max_tavily_calls = max(0, int(max_tavily_calls))
        self.max_llm_calls = max(0, int(max_llm_calls))
        self.tavily_calls_used = 0
        self.llm_calls_used = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "tavily_calls_used": int(self.tavily_calls_used),
            "tavily_calls_limit": int(self.max_tavily_calls),
            "llm_calls_used": int(self.llm_calls_used),
            "llm_calls_limit": int(self.max_llm_calls),
        }

    def consume_tavily(self, *, outcome: str, callsite: str, count: int = 1) -> None:
        self._consume(
            service="tavily",
            outcome=outcome,
            callsite=callsite,
            count=count,
        )

    def consume_llm(self, *, outcome: str, callsite: str, count: int = 1) -> None:
        self._consume(
            service="llm",
            outcome=outcome,
            callsite=callsite,
            count=count,
        )

    def _consume(
        self,
        *,
        service: str,
        outcome: str,
        callsite: str,
        count: int,
    ) -> None:
        clean_service = str(service or "").strip().lower()
        clean_outcome = str(outcome or "").strip()
        clean_callsite = str(callsite or "").strip()
        requested = max(1, int(count))

        if clean_service == "tavily":
            used = int(self.tavily_calls_used)
            limit = int(self.max_tavily_calls)
        else:
            used = int(self.llm_calls_used)
            limit = int(self.max_llm_calls)

        if used + requested > limit:
            raise CycleBudgetExhausted(
                service=clean_service,
                outcome=clean_outcome,
                callsite=clean_callsite,
                used=used,
                limit=limit,
                snapshot=self.snapshot(),
            )

        if clean_service == "tavily":
            self.tavily_calls_used += requested
        else:
            self.llm_calls_used += requested
