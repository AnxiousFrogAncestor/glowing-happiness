from fastapi import APIRouter, Query
from typing import List, Dict
from .finance_api import gather_links

router = APIRouter()

@router.get("/company-news", response_model=List[Dict[str, str]])
def get_company_news(company: str = Query(..., description="Company name or ticker to fetch news for")):
    return gather_links(company)
