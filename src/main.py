from fastapi import FastAPI
from .news_router import router as news_router
from .finance_router import router as finance_router
from .cluster_router import router as cluster_router
from .model_router import router as model_router

app = FastAPI()
# from the root folder
# uvicorn src.main:app --reload
app.include_router(news_router, prefix="/api/v1/news", tags=["news"])
app.include_router(finance_router, prefix="/api/v1/finance", tags=["finance"])
app.include_router(cluster_router, prefix="/api/v1/cluster", tags=["cluster"])
app.include_router(model_router, prefix="/api/v1/model", tags=["model"])
@app.get("/")
def read_root():
    return {"message":"Get the latest financial data."}

