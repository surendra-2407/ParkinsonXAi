"""
database.py — Async MongoDB connection manager using Motor.
Connects once at FastAPI startup, exposes db collections globally.
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

# Global references set during lifespan startup
client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None


async def connect_db(mongo_url: str, db_name: str = "parkinson_xai") -> None:
    """Initialize Motor client and create all indexes."""
    global client, db

    logger.info("Connecting to MongoDB...")
    client = AsyncIOMotorClient(
        mongo_url,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
    )

    # Ping to verify connection
    await client.admin.command("ping")
    db = client[db_name]
    logger.info(f"Connected to MongoDB — database: '{db_name}'")

    await _create_indexes()


async def close_db() -> None:
    """Close Motor client on shutdown."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed.")


async def _create_indexes() -> None:
    """Create all collection indexes idempotently."""
    global db

    # ── predictions collection ──────────────────────────────────────────
    pred_col = db["predictions"]
    await pred_col.create_indexes([
        IndexModel([("timestamp", DESCENDING)], name="idx_timestamp_desc"),
        IndexModel([("session_id", ASCENDING)], name="idx_session_id"),
        IndexModel([("detection.label", ASCENDING)], name="idx_detection_label"),
    ])

    # ── model_stats collection ──────────────────────────────────────────
    stats_col = db["model_stats"]
    await stats_col.create_indexes([
        IndexModel([("updated_at", DESCENDING)], name="idx_updated_at_desc"),
    ])

    # ── sessions collection ─────────────────────────────────────────────
    sess_col = db["sessions"]
    await sess_col.create_indexes([
        IndexModel([("session_id", ASCENDING)], unique=True, name="idx_session_unique"),
        IndexModel([("last_active", DESCENDING)], name="idx_last_active"),
    ])

    logger.info("MongoDB indexes ensured for all collections.")


def get_db() -> AsyncIOMotorDatabase:
    """FastAPI dependency: returns the active database handle (may be None)."""
    return db


def is_connected() -> bool:
    """Returns True if MongoDB is connected."""
    return db is not None


def get_predictions_col():
    if db is None:
        return None
    return db["predictions"]


def get_stats_col():
    if db is None:
        return None
    return db["model_stats"]


def get_sessions_col():
    if db is None:
        return None
    return db["sessions"]
