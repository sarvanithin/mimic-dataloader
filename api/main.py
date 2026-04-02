"""
MIMIC DataLoader API — x402/MPP-protected REST wrapper.

Exposes dataset metadata, schema info, and sample record previews.
Requires credentialed MIMIC-IV access for full data; demo endpoints return schema only.

Paid endpoints:
  POST /v1/schema     $0.01 — get column schema for a MIMIC task
  POST /v1/stats      $0.01 — get dataset statistics for a task
  POST /v1/preview    $0.02 — get 5 anonymized sample rows for a task

Free endpoints:
  GET  /              redirect to /docs
  GET  /health        service health
  GET  /docs          OpenAPI UI
  GET  /v1/tasks      list available tasks (no payment needed)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ---------------------------------------------------------------------------
# x402 middleware
# ---------------------------------------------------------------------------

ENDPOINT_PRICES: dict[str, str] = {
    "/v1/schema": "0.01",
    "/v1/stats": "0.01",
    "/v1/preview": "0.02",
}

FREE_PATHS: set[str] = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/v1/tasks"}

_SECRET = os.getenv("MPP_SECRET_KEY", "mimic-dev-secret")


def _valid_payment(token: str, path: str) -> bool:
    minute = str(int(time.time()) // 60)
    for window in (minute, str(int(minute) - 1)):
        expected = hmac.new(
            _SECRET.encode(),
            f"{path}:{window}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected, token):
            return True
    if hmac.compare_digest(_SECRET, token):
        return True
    return False


class X402Middleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in FREE_PATHS:
            return await call_next(request)
        if request.method != "POST" or path not in ENDPOINT_PRICES:
            return await call_next(request)

        auth = request.headers.get("X-PAYMENT") or request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()

        if token and _valid_payment(token, path):
            response = await call_next(request)
            response.headers["X-PAYMENT-ACCEPTED"] = "true"
            return response

        price = ENDPOINT_PRICES[path]
        return JSONResponse(
            status_code=402,
            content={
                "error": "Payment required",
                "path": path,
                "price_usd": price,
                "protocol": "x402",
                "instructions": "Send X-PAYMENT: Bearer mimic-dev-secret for demo access.",
            },
            headers={"X-PAYMENT-REQUIRED": "true", "X-PRICE-USD": price, "X-PROTOCOL": "x402"},
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MIMIC DataLoader API",
    description=(
        "x402-protected REST API for MIMIC-IV clinical dataset schemas, statistics, "
        "and sample previews. Full data access requires credentialed PhysioNet access."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(X402Middleware)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

Task = Literal["mortality", "readmission", "los", "sepsis", "phenotyping", "decompensation"]

TASK_SCHEMAS: dict[str, dict] = {
    "mortality": {
        "description": "In-hospital mortality prediction (48-hour window)",
        "features": ["age", "gender", "admission_type", "vitals_ts", "labs_ts", "icd_codes"],
        "label": "binary (died_in_hospital)",
        "rows_train": 14681, "rows_val": 1836, "rows_test": 2286,
        "seq_length": 48,
    },
    "readmission": {
        "description": "30-day readmission prediction",
        "features": ["age", "gender", "discharge_location", "diagnoses", "procedures"],
        "label": "binary (readmitted_30d)",
        "rows_train": 32109, "rows_val": 4014, "rows_test": 4014,
        "seq_length": 48,
    },
    "los": {
        "description": "Length of stay prediction (regression, days)",
        "features": ["age", "gender", "admission_type", "vitals_ts", "labs_ts"],
        "label": "continuous (los_days)",
        "rows_train": 29105, "rows_val": 3638, "rows_test": 3638,
        "seq_length": 48,
    },
    "sepsis": {
        "description": "Sepsis onset prediction (6-hour early warning)",
        "features": ["vitals_ts", "labs_ts", "antibiotics", "cultures"],
        "label": "binary (sepsis_onset)",
        "rows_train": 8843, "rows_val": 1105, "rows_test": 1105,
        "seq_length": 72,
    },
    "phenotyping": {
        "description": "25-class ICD phenotype classification",
        "features": ["clinical_notes", "icd_codes", "labs_ts"],
        "label": "multi-label (25 phenotypes)",
        "rows_train": 29249, "rows_val": 3656, "rows_test": 3656,
        "seq_length": None,
    },
    "decompensation": {
        "description": "Physiologic decompensation prediction (hourly)",
        "features": ["vitals_ts", "labs_ts", "gcs"],
        "label": "binary (decompensated_next_24h)",
        "rows_train": 2377768, "rows_val": 297221, "rows_test": 297221,
        "seq_length": 24,
    },
}


class TaskRequest(BaseModel):
    task: Task
    split: str = "train"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "mimic-dataloader-api", "version": "0.1.0"}


@app.get("/v1/tasks")
async def list_tasks():
    """List all available MIMIC-IV prediction tasks (free)."""
    return {
        "tasks": list(TASK_SCHEMAS.keys()),
        "note": "Full dataset access requires credentialed PhysioNet MIMIC-IV access.",
    }


@app.post("/v1/schema")
async def get_schema(body: TaskRequest):
    """Return column schema and metadata for a MIMIC-IV task. ($0.01)"""
    schema = TASK_SCHEMAS.get(body.task)
    if not schema:
        return JSONResponse(status_code=404, content={"error": f"Unknown task: {body.task}"})
    return {"task": body.task, "split": body.split, "schema": schema}


@app.post("/v1/stats")
async def get_stats(body: TaskRequest):
    """Return class balance and feature stats for a task. ($0.01)"""
    schema = TASK_SCHEMAS.get(body.task)
    if not schema:
        return JSONResponse(status_code=404, content={"error": f"Unknown task: {body.task}"})
    split_rows = {
        "train": schema.get("rows_train"),
        "val": schema.get("rows_val"),
        "test": schema.get("rows_test"),
    }
    return {
        "task": body.task,
        "split": body.split,
        "total_rows": split_rows.get(body.split, "N/A"),
        "feature_count": len(schema["features"]),
        "label_type": schema["label"],
        "seq_length": schema["seq_length"],
    }


@app.post("/v1/preview")
async def get_preview(body: TaskRequest):
    """Return 5 anonymized schema-level sample records (no real patient data). ($0.02)"""
    schema = TASK_SCHEMAS.get(body.task)
    if not schema:
        return JSONResponse(status_code=404, content={"error": f"Unknown task: {body.task}"})
    sample = {f: f"<{f}_value>" for f in schema["features"]}
    return {
        "task": body.task,
        "split": body.split,
        "note": "Sample records are schema-level only. Real data requires PhysioNet credentials.",
        "samples": [{"id": i, **sample, "label": "<label_value>"} for i in range(5)],
    }
