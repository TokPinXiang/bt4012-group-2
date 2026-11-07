"""Pydantic contracts for every hand-off in the agent pipeline.

    predictions_<model>.csv  --PredictionRow-->  alert report  --AlertReport-->
    LLM answer  --LLMNarrative-->  case report  --CaseReportFile-->  evaluation

Members 2 and 3 produce PredictionRow files. Everything else is produced and checked here.
Export JSON Schemas for the team with:  python -m src.agent.schemas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

TX_ID_PATTERN = r"^HI-Small:\d{8}$"
ACCOUNT_PATTERN = r"^[0-9]+:[0-9A-F]+$"

TxId = Annotated[str, StringConstraints(pattern=TX_ID_PATTERN)]
AccountId = Annotated[str, StringConstraints(pattern=ACCOUNT_PATTERN)]
TxAlias = Annotated[str, StringConstraints(pattern=r"^T\d+$")]
AccountAlias = Annotated[str, StringConstraints(pattern=r"^A\d+$")]
Split = Literal["train", "validation", "test"]
Timestamp = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")]
Typology = Literal["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "STACK",
                   "BIPARTITE", "RANDOM", "NONE"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- input from Members 2 and 3

class PredictionRow(Strict):
    """One scored transaction in predictions_<model>.csv."""

    tx_id: TxId = Field(description="Min Gu's transaction id, unchanged, e.g. HI-Small:03970858")
    split: Split
    score: float = Field(ge=0, le=1, allow_inf_nan=False, description="Probability of laundering; higher = more suspicious")
    predicted_label: Literal[0, 1] = Field(description="1 if score >= the threshold tuned on validation")
    model_name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]+$", min_length=1, max_length=40)]


# --------------------------------------------------------------------------- evidence (alert report)

class Transaction(Strict):
    tx_id: TxId
    ts: Timestamp
    src: AccountId
    dst: AccountId
    from_bank: str
    to_bank: str
    amount_paid: float = Field(ge=0)
    payment_currency: str
    amount_received: float = Field(ge=0)
    receiving_currency: str
    payment_format: str


class Edge(Transaction):
    hop: int = Field(ge=0)


class Window(Strict):
    start: Timestamp
    end: Timestamp


class Side(Strict):
    tx_count: int = Field(ge=0)
    counterparties: int = Field(ge=0)
    first_ts: Timestamp | None
    last_ts: Timestamp | None
    amount_by_currency: dict[str, float]


class AccountHistory(Strict):
    account_id: AccountId
    window: Window
    first_seen: Timestamp | None
    lifetime_tx_count: int = Field(ge=0)
    outgoing: Side
    incoming: Side
    window_tx_count: int = Field(ge=0)
    recent_transactions: list[Transaction]
    recent_transactions_truncated: bool


class Subgraph(Strict):
    hops: int
    window: Window
    accounts: list[AccountId]
    edges: list[Edge]
    truncated: bool


class AccountSignals(Strict):
    outgoing_tx: int
    incoming_tx: int
    distinct_recipients: int
    distinct_senders: int


class StructureSignals(Strict):
    window: Window
    source_account: AccountSignals
    destination_account: AccountSignals
    pair_history: dict[Literal["same_direction_before", "reverse_direction_before"], int]


class Cycle(Strict):
    length: int = Field(ge=2)
    accounts: list[AccountId]
    tx_ids: list[TxId]
    start_ts: Timestamp
    end_ts: Timestamp
    repeat_count: int = Field(ge=1)


class AlertMeta(Strict):
    rank: int = Field(ge=1)
    tx_id: TxId
    score: float = Field(ge=0, le=1)
    predicted_label: Literal[0, 1]
    model_name: str
    split: Split


# --------------------------------------------------------------------------- LLM answer (uses aliases)

class FanOutCheck(Strict):
    kind: Literal["fan_out"]
    account: AccountAlias = Field(description="Sender of every cited transaction")
    recipients: int = Field(ge=1, description="Exact number of distinct receivers among the cited transactions")


class FanInCheck(Strict):
    kind: Literal["fan_in"]
    account: AccountAlias = Field(description="Receiver of every cited transaction")
    senders: int = Field(ge=1, description="Exact number of distinct senders among the cited transactions")


class CycleCheck(Strict):
    kind: Literal["cycle"]
    accounts: list[AccountAlias] = Field(min_length=3, description="Route in time order; must end where it starts")


class ChainCheck(Strict):
    kind: Literal["chain"]
    accounts: list[AccountAlias] = Field(min_length=2, description="Route in time order")


class InvolvesCheck(Strict):
    kind: Literal["involves"]
    account: AccountAlias = Field(description="Every cited transaction has this account as sender or receiver")


class CountCheck(Strict):
    kind: Literal["count"]
    n: int = Field(ge=1, description="Exact number of cited transactions")


class TotalCheck(Strict):
    kind: Literal["total"]
    amount: float = Field(gt=0, description="Sum of the cited amounts (1% tolerance)")
    currency: str = Field(description="Every cited transaction must be in this currency, e.g. US Dollar")
    side: Literal["paid", "received"] = "paid"


class AmountRangeCheck(Strict):
    kind: Literal["amount_range"]
    min: float = Field(ge=0)
    max: float = Field(gt=0)


class WithinHoursCheck(Strict):
    kind: Literal["within_hours"]
    hours: float = Field(gt=0, description="All cited transactions happen within this many hours of each other")


Check = Annotated[
    Union[FanOutCheck, FanInCheck, CycleCheck, ChainCheck, InvolvesCheck, CountCheck, TotalCheck,
          AmountRangeCheck, WithinHoursCheck],
    Field(discriminator="kind"),
]


class LLMClaim(Strict):
    text: str = Field(min_length=1, description="One factual statement, using aliases like A1 and T4")
    cited_tx: list[TxAlias] = Field(min_length=1, description="Transactions (T...) the statement is about")
    checks: list[Check] = Field(min_length=1, description="Machine-checkable facts about the cited transactions")


class LLMNarrative(Strict):
    """The investigator's final answer. Submitted through the submit_case_report tool."""

    typology: Typology
    suspicious: bool
    confidence: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, description="2-3 sentences for a human investigator, using aliases")
    claims: list[LLMClaim] = Field(min_length=1, max_length=8)


# --------------------------------------------------------------------------- case report (real ids)

class Claim(Strict):
    text: str
    cited_tx_ids: list[TxId] = Field(min_length=1)
    checks: list[Check]


class DroppedClaim(Strict):
    text: str
    cited_tx_ids: list[str] = Field(description="Real ids, or the alias as written if it was made up")
    checks: list[Check]
    reasons: list[str]


class Narrative(Strict):
    typology: Typology
    suspicious: bool
    confidence: Literal["low", "medium", "high"]
    summary: str
    claims: list[Claim]


class Attempt(Strict):
    status: Literal["verified", "partial", "failed", "invalid"]
    claims_passed: int = 0
    claims_total: int = 0
    detail: str | None = None


class Verification(Strict):
    status: Literal["verified", "partial", "failed"]
    typology_supported: bool
    typology_reason: str | None
    claims_total: int
    claims_passed: int
    dropped_claims: list[DroppedClaim]
    attempts: list[Attempt]


class Triage(Strict):
    decision: Literal["fast_track", "manual_review", "auto_clear"]
    reason: str


class ToolCallLog(Strict):
    tool: str
    args: dict
    reason: str | None = Field(default=None, description="The agent's own one-sentence reason for the call")
    rows: int = 0
    new_transactions: int = 0
    error: str | None = None


class LLMRun(Strict):
    agent: str
    model: str
    usage: dict[str, int]
    tool_calls: list[ToolCallLog]


class CaseAliases(Strict):
    accounts: dict[AccountAlias, AccountId]
    transactions: dict[TxAlias, TxId]


class AlertReport(Strict):
    """Evidence for one alert, plus the slots the agent fills in."""

    alert: AlertMeta
    as_of: Timestamp | None = None
    flagged_transaction: Transaction | None = None
    accounts: dict[Literal["source", "destination"], AccountHistory] | None = None
    structure_signals: StructureSignals | None = None
    cycles_closed_by_flagged: list[Cycle] = []
    subgraph: Subgraph | None = None
    evidence_tx_ids: list[TxId] = []
    error: str | None = None
    # Filled by the agent.
    narrative: Narrative | None = None
    verification: Verification | None = None
    triage: Triage | None = None
    llm: LLMRun | None = None
    tool_transactions: list[Transaction] = Field(default=[], description="Extra transactions the agent fetched")
    case_aliases: CaseAliases | None = None


class CaseReportFile(Strict):
    report_version: Literal["alert-report/v1", "alert-report/v2"]
    generated_at: str
    source_predictions: str
    model_names: list[str]
    config: dict
    n_alerts: int
    n_errors: int
    llm_model: str | None = None
    agent: str | None = None
    verification_counts: dict[str, int] | None = None
    triage_counts: dict[str, int] | None = None
    llm_tokens: dict[str, int] | None = None
    alerts: list[AlertReport]


def export_json_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in (PredictionRow, AlertReport, LLMNarrative, CaseReportFile):
        path = out_dir / f"{model.__name__}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    for p in export_json_schemas(root / "docs" / "contracts"):
        print("wrote", p)
