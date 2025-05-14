from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError, root_validator
from .finance_api import get_fundamental_data

router = APIRouter()

class FinancialRequest(BaseModel):
    company_name: str = Field(..., description="Company name")
    income: bool = Field(True, description="Include income statement or not.")
    cash_flow: bool = Field(False, description="Include cash flow statement or not.")
    balance_sheet: bool = Field(False, description="Include balance sheet or not.")
    start_year: int = Field(..., description="Start year of the statement.")
    end_year: int = Field(..., description="End year of the statement.")

    @root_validator(skip_on_failure=True)
    def check_year_range(cls, values):
        start, end = values.get("start_year"), values.get("end_year")
        if start > end:
            raise ValueError("start_year must be less than or equal to end_year")
        return values


@router.post("/financials")
async def get_financials(request: FinancialRequest):
    """Returns the financial statement for a given company with the given flags."""
    result = {"company": request.company_name, "fin_statement": {}}

    if request.income:
        fund_data = get_fundamental_data(request.company_name, request.start_year, request.end_year, fundamental_data_type="income")
        result["fin_statement"]["income"] = fund_data
    elif request.balance_sheet:
        fund_data = get_fundamental_data(request.company_name, request.start_year, request.end_year, fundamental_data_type="balance")
        result["fin_statement"]["balance_sheet"] = fund_data

    return result