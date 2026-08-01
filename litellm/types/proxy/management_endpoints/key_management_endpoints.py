from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class BulkUpdateKeyRequestItem(BaseModel):
    """Individual key update request item"""

    key: str  # Key identifier (token)
    budget_id: Optional[str] = None  # Budget ID associated with the key
    max_budget: Optional[float] = None  # Max budget for key
    team_id: Optional[str] = None  # Team ID associated with key
    tags: Optional[List[str]] = None  # Tags for organizing keys


class BulkUpdateKeyRequest(BaseModel):
    """Request for bulk key updates"""

    keys: List[BulkUpdateKeyRequestItem]


class SuccessfulKeyUpdate(BaseModel):
    """Successfully updated key with its updated information"""

    key: str
    key_info: Dict[str, Any]


class FailedKeyUpdate(BaseModel):
    """Failed key update with reason"""

    key: str
    key_info: Optional[Dict[str, Any]] = None
    failed_reason: str


class BulkUpdateKeyResponse(BaseModel):
    """Response for bulk key update operations"""

    total_requested: int
    successful_updates: List[SuccessfulKeyUpdate]
    failed_updates: List[FailedKeyUpdate]


class EligibleKeyRequest(BaseModel):
    """Request the complete set of eligible keys for one or more user identities."""

    user_ids: List[str]


class EligibleKeySummary(BaseModel):
    """Minimal key fields required to select and display an eligible key."""

    key_name: Optional[str] = None
    key_alias: Optional[str] = None
    spend: float
    max_budget: Optional[float] = None
    expires: Optional[datetime] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    blocked: Optional[bool] = None
    availability: Literal[
        "available", "waiting_for_reset", "reset_pending", "exhausted"
    ]
    usable_now: bool


class EligibleKeyResponse(BaseModel):
    keys: List[EligibleKeySummary]
