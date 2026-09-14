# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from artifice_transcribe.config import settings
from artifice_transcribe.db.models import Base
from artifice_transcribe.db.session import get_db
from artifice_transcribe.main import app

# test_api.py is a standalone live-server verification script, not a pytest
# module: it expects a running server on 127.0.0.1:8000 and takes the audio
# path as sys.argv[1]. Collected as a test module it can only fail with
# "Connection refused". It stays runnable as `python tests/test_api.py
# [audio]`; its no-live-engine coverage lives in tests/test_api_e2e.py.
collect_ignore = ["test_api.py"]


@dataclass
class ApiFixture:
    client: AsyncClient
    session_factory: async_sessionmaker[AsyncSession]
    upload_dir: Path


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    """An AsyncClient + DB session factory wired to an isolated sqlite file
    and upload dir per test.

    Overrides both ``get_db`` (the FastAPI dependency) and the module-level
    ``async_session`` factory so that background tasks also use the test
    database instead of the real one.
    """
    db_path = tmp_path / "test.db"
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    test_session = async_sessionmaker(test_engine, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # Also override the module-level session factory so background tasks
    # (which open their own sessions via ``async_session()``) use the
    # isolated test database.
    monkeypatch.setattr(
        "artifice_transcribe.api.v1.routes.async_session",
        test_session,
    )

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(settings, "upload_dir", str(upload_dir))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ApiFixture(client=ac, session_factory=test_session, upload_dir=upload_dir)

    app.dependency_overrides.clear()
    await test_engine.dispose()
