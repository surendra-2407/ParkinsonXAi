"""
setup_mongodb.py — One-time script to create all MongoDB collections and indexes.
Run once after setting MONGODB_URL in .env:
    cd backend
    python setup_mongodb.py
"""
import os, asyncio, sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MONGO_URL = os.getenv("MONGODB_URL", "")
DB_NAME   = os.getenv("DATABASE_NAME", "parkinson_xai")

if not MONGO_URL or MONGO_URL == "PASTE_YOUR_MONGODB_ATLAS_URL_HERE":
    print("❌  MONGODB_URL not set in .env")
    sys.exit(1)

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING

async def setup():
    print("🔗  Connecting to MongoDB Atlas...")
    client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=10000)
    await client.admin.command("ping")
    print(f"✅  Connected — database: '{DB_NAME}'")
    db = client[DB_NAME]

    # ── predictions ────────────────────────────────────────────────────
    pred = db["predictions"]
    await pred.create_indexes([
        IndexModel([("timestamp", DESCENDING)],      name="idx_timestamp_desc"),
        IndexModel([("session_id", ASCENDING)],       name="idx_session_id"),
        IndexModel([("detection.label", ASCENDING)],  name="idx_detection_label"),
        IndexModel([("prediction_type", ASCENDING)],  name="idx_prediction_type"),
    ])
    print("✅  predictions  — 4 indexes created")

    # ── model_stats ────────────────────────────────────────────────────
    stats = db["model_stats"]
    await stats.create_indexes([
        IndexModel([("updated_at", DESCENDING)], name="idx_updated_at_desc"),
    ])
    print("✅  model_stats  — 1 index created")

    # ── sessions ───────────────────────────────────────────────────────
    sess = db["sessions"]
    await sess.create_indexes([
        IndexModel([("session_id", ASCENDING)], unique=True, name="idx_session_unique"),
        IndexModel([("last_active", DESCENDING)],             name="idx_last_active"),
    ])
    print("✅  sessions     — 2 indexes created")

    # Show collection list
    cols = await db.list_collection_names()
    print(f"\n📦  Collections in '{DB_NAME}': {cols or '(will be created on first write)'}")

    # Insert bootstrap model_stats document (idempotent)
    now = datetime.now(timezone.utc)
    existing = await stats.find_one({"_id": "global"})
    if not existing:
        await stats.insert_one({
            "_id": "global",
            "total_predictions": 0,
            "parkinson_count": 0,
            "healthy_count": 0,
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0,
            "updated_at": now,
        })
        print("✅  model_stats  — bootstrap document inserted")
    else:
        print("ℹ️   model_stats  — bootstrap document already exists, skipping")

    client.close()
    print("\n🎉  MongoDB setup complete!")
    print(f"    Database   : {DB_NAME}")
    print(f"    Collections: predictions · model_stats · sessions")

asyncio.run(setup())
