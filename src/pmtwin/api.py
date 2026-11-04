"""HTTP API of the twin service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pmtwin.config import Settings
from pmtwin.predictors import GbmPredictor
from pmtwin.schemas import (
    FleetItem,
    HealthResponse,
    HistoryPoint,
    IngestRequest,
    IngestResponse,
    TwinSnapshot,
)
from pmtwin.store import TwinStore
from pmtwin.twin import CycleConflict, ModelUnavailable, TwinRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    store = TwinStore(settings.db_path)
    predictor = GbmPredictor.load(settings.gbm_path) if settings.gbm_path.exists() else None
    app.state.registry = TwinRegistry(store, predictor, settings)
    yield
    store.close()


app = FastAPI(title="Predictive Maintenance Twin", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def reject_bad_input(request: Request, error: RequestValidationError) -> JSONResponse:
    """Leave the rejected input out: an infinite value cannot be written back as JSON."""
    details = [{key: item[key] for key in ("type", "loc", "msg")} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


def get_registry(request: Request) -> TwinRegistry:
    return request.app.state.registry


RegistryDep = Annotated[TwinRegistry, Depends(get_registry)]
EngineId = Annotated[int, Path(ge=1)]


@app.get("/health")
def health(registry: RegistryDep) -> HealthResponse:
    if registry.predictor is None:
        return HealthResponse(status="no_model", models={})
    return HealthResponse(status="ok", models={"rul": registry.predictor.name})


@app.post("/ingest")
def ingest(request: IngestRequest, registry: RegistryDep) -> IngestResponse:
    try:
        return IngestResponse(snapshots=registry.ingest(request.readings))
    except CycleConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail="No trained model is loaded.") from error


@app.get("/fleet")
def fleet(registry: RegistryDep) -> list[FleetItem]:
    return registry.fleet()


@app.get("/engines/{engine_id}")
def engine(engine_id: EngineId, registry: RegistryDep) -> TwinSnapshot:
    snapshot = registry.snapshot(engine_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} is unknown.")
    return snapshot


@app.get("/engines/{engine_id}/history")
def engine_history(
    engine_id: EngineId,
    registry: RegistryDep,
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> list[HistoryPoint]:
    if registry.snapshot(engine_id) is None:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} is unknown.")
    return registry.history(engine_id, limit)
