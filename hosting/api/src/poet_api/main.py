# Start the server (from hosting/api):
#   uvicorn poet_api.main:app --reload
# Interactive API docs once running:
#   Swagger UI -> http://127.0.0.1:8000/docs
#   ReDoc      -> http://127.0.0.1:8000/redoc

from fastapi import FastAPI

from poet_api.routers import health

app = FastAPI()
app.include_router(health.router)
