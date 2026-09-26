from fastapi import APIRouter
from typing import List
from api.schemas.policy import PolicyInfo
from api.services.policy_service import get_available_policies

router = APIRouter(prefix="/policies", tags=["policies"])

@router.get("", response_model=List[PolicyInfo])
def list_policies():
    """List all available policies and their readiness state."""
    return get_available_policies()
