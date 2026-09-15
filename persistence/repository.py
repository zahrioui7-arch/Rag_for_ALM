import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from persistence.models import User, ChatSession, Message


async def get_or_create_user(db: AsyncSession, username: str) -> User:
    result = await db.execute(select(User).where(User.external_id == username))
    user = result.scalar_one_or_none()
    if user:
        return user
    user = User(external_id=username)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def create_session(db: AsyncSession, user_id: uuid.UUID, title: str = "New conversation") -> ChatSession:
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: uuid.UUID) -> ChatSession | None:
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    return result.scalar_one_or_none()


async def list_sessions(db: AsyncSession, user_id: uuid.UUID) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_messages(db: AsyncSession, session_id: uuid.UUID) -> list[Message]:
    result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def add_message(db: AsyncSession, session_id: uuid.UUID, role: str, content: str, extra: dict | None = None) -> Message:
    message = Message(session_id=session_id, role=role, content=content, extra=extra or {})
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def maybe_set_session_title(db: AsyncSession, session: ChatSession, first_user_message: str) -> None:
    """Auto-title a session from its first message, once."""
    if session.title == "New conversation":
        session.title = first_user_message[:80]
        db.add(session)
        await db.commit()