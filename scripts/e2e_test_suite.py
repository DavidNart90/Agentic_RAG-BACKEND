import asyncio
import json
import uuid
import logging
from typing import List, Dict, Any
import argparse
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from httpx import AsyncClient, ASGITransport

from src.trackrealties.api.main import app
from src.trackrealties.api import dependencies as dep
from src.trackrealties.api.routes import agents as agents_routes
from src.trackrealties.core.database import db_pool
from src.trackrealties.data import repository as repo_module
from src.trackrealties.models.db import Session, UserRole, ConversationMessage, MessageRole


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- Fake repository implementations ----
class FakeSessionRepository:
    def __init__(self, conn):
        self.conn = conn

    async def create_user_session(
        self,
        role: UserRole,
        user_id: str | None = None,
        session_data: Dict[str, Any] | None = None,
        timeout_minutes: int = 60,
    ) -> Session:
        now = datetime.tcnouw()
        return Session(
            id=uuid.uuid4(),
            user_id=user_id,
            user_role=role,
            session_data=session_data or {},
            created_at=now,
            updated_at=now,
            expires_at=now,
            is_active=True,
        )

    async def get_session(self, session_id: uuid.UUID) -> Session | None:
        now = datetime.now(timezone.utc) 
        return Session(
            id=session_id,
            user_id="test",
            user_role=UserRole.BUYER,
            session_data={},
            created_at=now,
            updated_at=now,
            expires_at=now,
            is_active=True,
        )


class FakeMessageRepository:
    def __init__(self, conn):
        self.conn = conn

    async def add_conversation_message(
        self,
        session_id: uuid.UUID,
        role: MessageRole,
        content: str,
        **kwargs,
    ) -> ConversationMessage:
        return ConversationMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role=role,
            content=content,
            tools_used=kwargs.get("tools_used", []),
            validation_result=None,
            confidence_score=kwargs.get("confidence_score"),
            processing_time_ms=None,
            token_count=None,
            message_metadata=kwargs.get("message_metadata", kwargs.get("metadata", {})),
        )


# Patch repository classes
repo_module.SessionRepository = FakeSessionRepository
repo_module.MessageRepository = FakeMessageRepository
agents_routes.SessionRepository = FakeSessionRepository
agents_routes.MessageRepository = FakeMessageRepository

# Override database connection dependency to avoid real DB calls
@asynccontextmanager
async def _dummy_connection():
    class Dummy:
        async def fetch(self, *args, **kwargs):
            return []

        async def fetchrow(self, *args, **kwargs):
            return None

        async def execute(self, *args, **kwargs):
            return None

        async def fetchval(self, *args, **kwargs):
            return None

    yield Dummy()

app.dependency_overrides[dep.get_db_connection] = lambda: _dummy_connection()
db_pool.initialize = AsyncMock()
db_pool.close = AsyncMock()
db_pool.acquire = _dummy_connection


async def ingest_sample_data() -> None:
    """Simulate ingestion of sample data."""
    with open("examples/sample_market_data.json", "r") as f:
        market = json.load(f)
    with open("examples/sample_property_listings.json", "r") as f:
        listings = json.load(f)
    logger.info("Ingested %d market records and %d listings", len(market), len(listings))


queries = {
    "investor": "What is the ROI on 24727 Bogey Rdg, San Antonio, TX?",
    "developer": "What is the median price in Athens, TX?",
    "buyer": "Who is the listing agent for 333 Florida St, San Antonio, TX 78210?",
    "agent": "Tell me about 7 W 21st St, New York, NY 10010",
}


async def call_chat_endpoint(client: AsyncClient, role: str) -> Dict[str, Any]:
    payload = {"message": queries[role], "session_id": str(uuid.uuid4())}
    resp = await client.post(f"/agents/{role}/chat", json=payload)
    try:
        body = resp.json()
    except Exception:
        logger.error("Failed to decode JSON for role %s: %s", role, resp.text)
        body = {"error": "invalid_json", "raw": resp.text}
    return {"status": resp.status_code, "body": body}


async def run_suite(retries: int) -> List[Dict[str, Any]]:
    results = []
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await ingest_sample_data()
        for endpoint in ["/health"]:
            r = await client.get(endpoint)
            results.append({"endpoint": endpoint, "status": r.status_code})

        for role in ["investor", "developer", "buyer", "agent"]:
            attempts = 0
            success = False
            while attempts <= retries and not success:
                res = await call_chat_endpoint(client, role)
                success = res["status"] == 200 and res["body"].get("assistant_message_id")
                attempts += 1
            res["attempts"] = attempts
            res["role"] = role
            results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output", type=str, help="Path to save JSON results")
    args = parser.parse_args()
    results = asyncio.run(run_suite(args.retries))
    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
    print(output)


if __name__ == "__main__":
    main()
