from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.routes import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.core.model_loader import get_model
from fastapi import Request, Response

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # preload model
    yield

app = FastAPI(title="Reasoning Engine API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["127.0.0.1"]
)

# Register routes
app.include_router(router)

@app.middleware("http")
async def ignore_well_known(request: Request, call_next):
    if request.url.path.startswith("/.well-known/"):
        return Response(status_code=204)
    return await call_next(request)