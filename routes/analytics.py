from fastapi import APIRouter
from datetime import datetime, timedelta, timezone
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Claude Sonnet 4.6 pricing per token
INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000   # $3.00 / M tokens
OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000   # $15.00 / M tokens


# ─── helpers ─────────────────────────────────────────────────────────────────

def _last_n_days(n: int):
    now = datetime.now(timezone.utc)
    return now - timedelta(days=n - 1)


def _day_bucket_pipeline(date_field: str, n_days: int):
    """Return aggregation stages that group by calendar day (UTC)."""
    since = _last_n_days(n_days)
    return [
        {"$match": {date_field: {"$gte": since}}},
        {"$group": {
            "_id": {
                "y": {"$year": f"${date_field}"},
                "m": {"$month": f"${date_field}"},
                "d": {"$dayOfMonth": f"${date_field}"},
            }
        }},
    ]


# ─── retail order analytics ───────────────────────────────────────────────────

@router.get("/orders")
async def order_analytics():
    # --- summary stats ---------------------------------------------------------
    pipeline_summary = [
        {"$group": {
            "_id": None,
            "total_orders":  {"$sum": 1},
            "total_revenue": {"$sum": "$total"},
            "avg_order_value": {"$avg": "$total"},
            "total_items_sold": {"$sum": {"$sum": "$items.quantity"}},
        }},
    ]
    summary_raw = await db.orders.aggregate(pipeline_summary).to_list(1)
    summary = summary_raw[0] if summary_raw else {
        "total_orders": 0, "total_revenue": 0,
        "avg_order_value": 0, "total_items_sold": 0,
    }
    summary.pop("_id", None)

    # --- daily revenue / orders (last 7 days) ----------------------------------
    since_7 = _last_n_days(7)
    pipeline_daily = [
        {"$match": {"placed_at": {"$gte": since_7}}},
        {"$group": {
            "_id": {
                "y": {"$year": "$placed_at"},
                "m": {"$month": "$placed_at"},
                "d": {"$dayOfMonth": "$placed_at"},
            },
            "orders":  {"$sum": 1},
            "revenue": {"$sum": "$total"},
        }},
        {"$sort": {"_id.y": 1, "_id.m": 1, "_id.d": 1}},
    ]
    daily_raw = await db.orders.aggregate(pipeline_daily).to_list(7)
    daily = [
        {
            "date": f"{r['_id']['y']}-{r['_id']['m']:02d}-{r['_id']['d']:02d}",
            "orders": r["orders"],
            "revenue": round(r["revenue"], 2),
        }
        for r in daily_raw
    ]

    # --- top items by quantity sold ---------------------------------------------
    pipeline_items = [
        {"$unwind": "$items"},
        {"$group": {
            "_id": "$items.name",
            "emoji":    {"$first": "$items.emoji"},
            "quantity": {"$sum": "$items.quantity"},
            "revenue":  {"$sum": {"$multiply": ["$items.price", "$items.quantity"]}},
        }},
        {"$sort": {"quantity": -1}},
        {"$limit": 10},
    ]
    top_items = await db.orders.aggregate(pipeline_items).to_list(10)
    for item in top_items:
        item["name"]    = item.pop("_id")
        item["revenue"] = round(item["revenue"], 2)

    # --- category revenue breakdown --------------------------------------------
    pipeline_category = [
        {"$unwind": "$items"},
        # join menu_items to get category
        {"$lookup": {
            "from": "menu_items",
            "localField": "items.id",
            "foreignField": "id",
            "as": "menu",
        }},
        {"$unwind": {"path": "$menu", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": {"$ifNull": ["$menu.category", "unknown"]},
            "revenue":  {"$sum": {"$multiply": ["$items.price", "$items.quantity"]}},
            "quantity": {"$sum": "$items.quantity"},
        }},
        {"$sort": {"revenue": -1}},
    ]
    category_raw = await db.orders.aggregate(pipeline_category).to_list(20)
    categories = [
        {"category": r["_id"], "revenue": round(r["revenue"], 2), "quantity": r["quantity"]}
        for r in category_raw
    ]

    # --- payment method distribution -------------------------------------------
    pipeline_payment = [
        {"$group": {
            "_id": "$payment.method",
            "count": {"$sum": 1},
        }},
        {"$sort": {"count": -1}},
    ]
    payment_raw = await db.orders.aggregate(pipeline_payment).to_list(10)
    payments = [{"method": r["_id"], "count": r["count"]} for r in payment_raw]

    # --- order status breakdown ------------------------------------------------
    pipeline_status = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    status_raw = await db.orders.aggregate(pipeline_status).to_list(10)
    statuses = [{"status": r["_id"], "count": r["count"]} for r in status_raw]

    return {
        "summary":    summary,
        "daily":      daily,
        "top_items":  top_items,
        "categories": categories,
        "payments":   payments,
        "statuses":   statuses,
    }


# ─── inference analytics ──────────────────────────────────────────────────────

@router.get("/inference")
async def inference_analytics():
    # --- summary ---------------------------------------------------------------
    pipeline_summary = [
        {"$group": {
            "_id": None,
            "total_requests":  {"$sum": 1},
            "total_input_tokens":  {"$sum": "$input_tokens"},
            "total_output_tokens": {"$sum": "$output_tokens"},
            "avg_latency_ms":  {"$avg": "$latency_ms"},
            "avg_tool_calls":  {"$avg": "$tool_call_count"},
            "total_tool_calls": {"$sum": "$tool_call_count"},
            "success_count":   {"$sum": {"$cond": ["$success", 1, 0]}},
        }},
    ]
    summary_raw = await db.inference_logs.aggregate(pipeline_summary).to_list(1)
    if summary_raw:
        s = summary_raw[0]
        s.pop("_id", None)
        total_in  = s.get("total_input_tokens", 0)
        total_out = s.get("total_output_tokens", 0)
        total_req = s.get("total_requests", 1)
        s["estimated_cost_usd"] = round(
            total_in * INPUT_COST_PER_TOKEN + total_out * OUTPUT_COST_PER_TOKEN, 4
        )
        s["success_rate"] = round(s["success_count"] / total_req * 100, 1)
        s["avg_latency_ms"] = round(s.get("avg_latency_ms") or 0, 0)
        s["avg_tool_calls"]  = round(s.get("avg_tool_calls") or 0, 2)
        summary = s
    else:
        summary = {
            "total_requests": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "avg_latency_ms": 0, "avg_tool_calls": 0, "total_tool_calls": 0,
            "success_count": 0, "estimated_cost_usd": 0, "success_rate": 100,
        }

    # --- daily request volume (last 7 days) ------------------------------------
    since_7 = _last_n_days(7)
    pipeline_daily = [
        {"$match": {"timestamp": {"$gte": since_7}}},
        {"$group": {
            "_id": {
                "y": {"$year": "$timestamp"},
                "m": {"$month": "$timestamp"},
                "d": {"$dayOfMonth": "$timestamp"},
            },
            "requests":      {"$sum": 1},
            "input_tokens":  {"$sum": "$input_tokens"},
            "output_tokens": {"$sum": "$output_tokens"},
            "avg_latency":   {"$avg": "$latency_ms"},
        }},
        {"$sort": {"_id.y": 1, "_id.m": 1, "_id.d": 1}},
    ]
    daily_raw = await db.inference_logs.aggregate(pipeline_daily).to_list(7)
    daily = [
        {
            "date": f"{r['_id']['y']}-{r['_id']['m']:02d}-{r['_id']['d']:02d}",
            "requests":      r["requests"],
            "input_tokens":  r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "avg_latency":   round(r["avg_latency"] or 0, 0),
        }
        for r in daily_raw
    ]

    # --- tool usage frequency --------------------------------------------------
    pipeline_tools = [
        {"$unwind": "$tool_calls"},
        {"$group": {
            "_id": "$tool_calls",
            "count": {"$sum": 1},
        }},
        {"$sort": {"count": -1}},
    ]
    tools_raw = await db.inference_logs.aggregate(pipeline_tools).to_list(20)
    tools = [{"tool": r["_id"], "count": r["count"]} for r in tools_raw]

    # --- latency distribution buckets ------------------------------------------
    pipeline_latency = [
        {"$bucket": {
            "groupBy": "$latency_ms",
            "boundaries": [0, 500, 1000, 2000, 4000, 8000, 99999],
            "default": "other",
            "output": {"count": {"$sum": 1}},
        }},
    ]
    latency_raw = await db.inference_logs.aggregate(pipeline_latency).to_list(10)
    labels = ["<0.5s", "0.5–1s", "1–2s", "2–4s", "4–8s", ">8s"]
    latency_dist = [
        {"bucket": labels[i] if i < len(labels) else str(r["_id"]), "count": r["count"]}
        for i, r in enumerate(latency_raw)
        if r.get("_id") != "other"
    ]

    # --- model usage (future-proof if multiple models logged) ------------------
    pipeline_models = [
        {"$group": {
            "_id": {"$ifNull": ["$model", "claude-sonnet-4-6"]},
            "count": {"$sum": 1},
            "input_tokens":  {"$sum": "$input_tokens"},
            "output_tokens": {"$sum": "$output_tokens"},
        }},
        {"$sort": {"count": -1}},
    ]
    models_raw = await db.inference_logs.aggregate(pipeline_models).to_list(10)
    models = [
        {
            "model": r["_id"],
            "requests": r["count"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": round(
                r["input_tokens"] * INPUT_COST_PER_TOKEN +
                r["output_tokens"] * OUTPUT_COST_PER_TOKEN, 4
            ),
        }
        for r in models_raw
    ]

    return {
        "summary":     summary,
        "daily":       daily,
        "tools":       tools,
        "latency_dist": latency_dist,
        "models":      models,
    }


# ─── combined overview ────────────────────────────────────────────────────────

@router.get("/overview")
async def analytics_overview():
    # lightweight combined summary for header cards
    order_sum = await db.orders.aggregate([
        {"$group": {
            "_id": None,
            "total_orders":  {"$sum": 1},
            "total_revenue": {"$sum": "$total"},
        }}
    ]).to_list(1)

    infer_sum = await db.inference_logs.aggregate([
        {"$group": {
            "_id": None,
            "total_requests": {"$sum": 1},
            "total_tokens":   {"$sum": {"$add": ["$input_tokens", "$output_tokens"]}},
        }}
    ]).to_list(1)

    o = order_sum[0] if order_sum else {"total_orders": 0, "total_revenue": 0}
    i = infer_sum[0] if infer_sum else {"total_requests": 0, "total_tokens": 0}

    return {
        "total_orders":     o["total_orders"],
        "total_revenue":    round(o["total_revenue"], 2),
        "total_ai_requests": i["total_requests"],
        "total_tokens":     i["total_tokens"],
    }
