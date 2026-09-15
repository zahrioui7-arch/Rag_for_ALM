import os
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Set ALM_DATABASE_URL env var in production; this default matches docker-compose.yml
DATABASE_URL = os.getenv(
    "ALM_DATABASE_URL",
    "postgresql+asyncpg://alm_user:alm_password@localhost:5432/alm_rag",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create tables if they don't exist. Call once on backend startup."""
    from persistence import models  # noqa: F401 -- ensures models are registered on Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session