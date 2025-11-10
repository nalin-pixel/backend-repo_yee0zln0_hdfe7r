import os
from io import BytesIO
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="ATC Smart Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "ATC Smart Pro Backend Running"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from ATC Smart Pro API"}

# Lazy import helpers to avoid startup failures if heavy libs are missing
_pd = None

def get_pd():
    global _pd
    if _pd is None:
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pandas not available: {e}")
        _pd = pd
    return _pd

# ---------- Utility: Column normalization ----------
COLUMN_ALIASES = {
    "priority": ["priority", "prio"],
    "check_title": ["check title", "check", "title", "rule", "check_name"],
    "check_message": ["check message", "message", "long text", "description"],
    "object_name": ["object name", "object", "obj_name"],
    "object_type": ["object type", "type", "obj_type"],
    "package": ["package", "package name", "pkg"],
}

REQUIRED_COLUMNS = [
    "priority",
    "check_title",
    "check_message",
    "object_name",
    "object_type",
    "package",
]


def normalize_columns(df: Any) -> Any:
    pd = get_pd()
    mapping: Dict[str, str] = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for std_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                mapping[lower_cols[alias]] = std_col
                break
    df = df.rename(columns=mapping)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    front = [c for c in REQUIRED_COLUMNS]
    rest = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return df[front + rest]

# ---------- Utility: Classification ----------
MANDATORY_KWS = [
    "obsolete", "removed", "not supported", "s/4hana", "hana", "mandatory",
    "addon not compatible", "must", "blocking", "error", "incompatible", "deprecated runtime",
]
OPTIONAL_KWS = [
    "performance", "optimize", "improve", "naming", "format", "style", "cleanup", "optional", "warning",
]
SYNTAX_KWS = [
    "syntax", "parser", "parse", "unknown statement", "unexpected", "missing", "typo", "keyword",
]

PRIORITY_WEIGHTS = {
    "1": 1.0, "very high": 1.0, "vh": 1.0,
    "2": 0.85, "high": 0.85, "h": 0.85,
    "3": 0.7, "medium": 0.7, "m": 0.7,
    "4": 0.55, "low": 0.55, "l": 0.55,
    "5": 0.4, "very low": 0.4, "vl": 0.4,
}


def score_text(text: str, keywords: List[str]) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in keywords if kw in t)
    if hits == 0:
        return 0.0
    return min(1.0, 0.55 + 0.25 * hits)


def classify_row(row: Any) -> Dict[str, Any]:
    title = str(row.get("check_title", ""))
    msg = str(row.get("check_message", ""))
    prio = str(row.get("priority", ""))
    base_m = max(score_text(title, MANDATORY_KWS), score_text(msg, MANDATORY_KWS))
    base_o = max(score_text(title, OPTIONAL_KWS), score_text(msg, OPTIONAL_KWS))
    base_s = max(score_text(title, SYNTAX_KWS), score_text(msg, SYNTAX_KWS))

    p = PRIORITY_WEIGHTS.get(prio.lower(), 0.6)
    m = base_m * (0.7 + 0.3 * p)
    o = base_o * (0.7 + 0.3 * (1 - abs(p - 0.6)))
    s = base_s * (0.7 + 0.3 * (1 - p))

    scores = {"Mandatory": m, "Optional": o, "Syntax": s}
    category = max(scores, key=scores.get)
    confidence = round(100 * scores[category], 1)
    return {"category": category, "confidence": confidence}


# ---------- Processing Endpoint ----------
@app.post("/api/process")
async def process_files(files: List[UploadFile] = File(...)):
    pd = get_pd()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    frames: List[Any] = []
    errors: List[str] = []

    for f in files:
        name = f.filename or "uploaded"
        try:
            content = await f.read()
            if name.lower().endswith((".xlsx", ".xls")):
                try:
                    df = pd.read_excel(BytesIO(content), engine="openpyxl")
                except Exception:
                    # Fallback without engine hint
                    df = pd.read_excel(BytesIO(content))
            elif name.lower().endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            else:
                errors.append(f"Unsupported file type: {name}")
                continue
            if df.empty:
                errors.append(f"No rows in {name}")
                continue
            df = normalize_columns(df)
            frames.append(df)
        except Exception as e:
            errors.append(f"Error reading {name}: {str(e)[:120]}")

    if not frames:
        raise HTTPException(status_code=400, detail="No valid data found in uploads")

    data = pd.concat(frames, ignore_index=True)

    # Apply classification
    cls = data.apply(classify_row, axis=1, result_type="expand")
    data["Category"] = cls["category"]
    data["Confidence"] = cls["confidence"]

    def to_safe(val):
        if hasattr(val, "__len__") and val == "nan":
            return ""
        try:
            import math
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return ""
        except Exception:
            pass
        return "" if val is None else str(val)

    records: List[Dict[str, Any]] = []
    for _, r in data.iterrows():
        rec = {
            "Priority": to_safe(r.get("priority", "")),
            "Check Title": to_safe(r.get("check_title", "")),
            "Check Message": to_safe(r.get("check_message", "")),
            "Object Name": to_safe(r.get("object_name", "")),
            "Object Type": to_safe(r.get("object_type", "")),
            "Package": to_safe(r.get("package", "")),
            "Category": to_safe(r.get("Category", "")),
            "Confidence": float(r.get("Confidence", 0.0)),
        }
        records.append(rec)

    total = len(records)
    by_cat = data["Category"].value_counts().to_dict()
    avg_conf = round(float(data["Confidence"].mean()), 1)

    return JSONResponse({
        "total": total,
        "by_category": by_cat,
        "avg_confidence": avg_conf,
        "records": records,
        "errors": errors,
    })


# ---------- Export Endpoint ----------
@app.post("/api/export")
async def export_excel(payload: Dict[str, Any]):
    pd = get_pd()
    records = payload.get("records")
    if not records:
        raise HTTPException(status_code=400, detail="No records provided for export")

    df = pd.DataFrame(records)

    summary = pd.DataFrame({
        "Category": ["Mandatory", "Optional", "Syntax"],
        "Count": [int((df["Category"] == c).sum()) for c in ["Mandatory", "Optional", "Syntax"]],
    })
    avg_conf = round(float(df["Confidence"].mean() if "Confidence" in df.columns and not df.empty else 0.0), 1)

    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Classified Findings")
            summary.to_excel(writer, index=False, sheet_name="Summary")
            meta = pd.DataFrame([
                {"Metric": "Total Findings", "Value": int(len(df))},
                {"Metric": "Average Confidence", "Value": avg_conf},
            ])
            meta.to_excel(writer, index=False, sheet_name="Metrics")
    except Exception:
        # Fallback to default engine if openpyxl not available
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer, index=False, sheet_name="Classified Findings")
            summary.to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="ATC_Smart_Pro_Categorized.xlsx"'}
    return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
