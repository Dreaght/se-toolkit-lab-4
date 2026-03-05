"""Unit tests for interaction edge cases and boundary values."""

from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from app.models.interaction import (
    InteractionLog,
    InteractionLogCreate,
    InteractionModel,
)
from app.models.learner import Learner  # noqa: F401 - registers learner table for FK
from app.routers.interactions import filter_by_max_item_id


def _make_log(
    id: int, learner_id: int, item_id: int, kind: str = "attempt"
) -> InteractionLog:
    """Helper to create an InteractionLog instance."""
    return InteractionLog(id=id, learner_id=learner_id, item_id=item_id, kind=kind)


def test_filter_with_zero_max_item_id() -> None:
    """Test filtering when max_item_id is zero - only item_id 0 or negative should pass."""
    interactions = [
        _make_log(1, 1, -1),
        _make_log(2, 1, 0),
        _make_log(3, 1, 1),
        _make_log(4, 1, 5),
    ]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=0)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[1].id == 2


def test_filter_with_negative_max_item_id() -> None:
    """Test filtering when max_item_id is negative."""
    interactions = [
        _make_log(1, 1, -5),
        _make_log(2, 1, -2),
        _make_log(3, 1, -1),
        _make_log(4, 1, 0),
        _make_log(5, 1, 1),
    ]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=-2)
    assert len(result) == 2
    assert result[0].id == 1
    assert result[1].id == 2


def test_filter_with_very_large_max_item_id() -> None:
    """Test filtering with a very large max_item_id - all interactions should pass."""
    interactions = [_make_log(i, 1, i) for i in range(1, 1001)]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=999999999)
    assert len(result) == 1000


def test_filter_preserves_original_order() -> None:
    """Test that filtering preserves the original order of interactions."""
    interactions = [
        _make_log(3, 1, 5),
        _make_log(1, 1, 2),
        _make_log(5, 1, 8),
        _make_log(2, 1, 3),
        _make_log(4, 1, 10),
    ]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=6)
    assert len(result) == 3
    assert result[0].id == 3
    assert result[1].id == 1
    assert result[2].id == 2


def test_filter_with_duplicate_item_ids() -> None:
    """Test filtering when multiple interactions have the same item_id."""
    interactions = [
        _make_log(1, 1, 3),
        _make_log(2, 2, 3),
        _make_log(3, 3, 3),
        _make_log(4, 1, 5),
        _make_log(5, 2, 5),
    ]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=3)
    assert len(result) == 3
    assert all(i.item_id == 3 for i in result)


def test_filter_with_negative_item_ids() -> None:
    """Test filtering interactions that have negative item_id values."""
    interactions = [
        _make_log(1, 1, -10),
        _make_log(2, 1, -5),
        _make_log(3, 1, 0),
        _make_log(4, 1, 5),
    ]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=-3)
    assert len(result) == 2
    assert result[0].item_id == -10
    assert result[1].item_id == -5


def test_filter_excludes_item_at_boundary_plus_one() -> None:
    """Test that interactions with item_id = max_item_id + 1 are excluded."""
    interactions = [
        _make_log(1, 1, 4),
        _make_log(2, 1, 5),
        _make_log(3, 1, 6),
    ]
    result = filter_by_max_item_id(interactions=interactions, max_item_id=5)
    assert len(result) == 2
    assert all(i.item_id <= 5 for i in result)
    assert not any(i.item_id == 6 for i in result)


@pytest.fixture
def db_session() -> Session:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Only create the tables we need for interaction testing
    # (avoid creating ItemRecord which uses PostgreSQL-specific JSONB)
    Learner.metadata.create_all(
        engine, tables=[Learner.__table__, InteractionLog.__table__]
    )
    with Session(engine) as session:
        yield session


def _create_interaction_sync(
    session: Session,
    learner_id: int,
    item_id: int,
    kind: str,
) -> InteractionLog:
    """Create a new interaction log in the database (synchronous version)."""
    interaction = InteractionLog(learner_id=learner_id, item_id=item_id, kind=kind)
    session.add(interaction)
    session.commit()
    session.refresh(interaction)
    return interaction


def _read_interactions_sync(session: Session) -> list[InteractionLog]:
    """Read all interactions from the database (synchronous version)."""
    result = session.exec(select(InteractionLog))
    return list(result.all())


def test_create_interaction_with_empty_kind(db_session: Session) -> None:
    """Test creating an interaction with an empty string kind."""
    interaction = _create_interaction_sync(
        session=db_session,
        learner_id=1,
        item_id=1,
        kind="",
    )
    assert interaction.kind == ""
    assert interaction.learner_id == 1
    assert interaction.item_id == 1


def test_create_interaction_with_special_characters_in_kind(
    db_session: Session,
) -> None:
    """Test creating an interaction with special characters in kind."""
    special_kinds = ["view!", "attempt#1", "complete%", "click_test", "scroll@home"]
    for kind in special_kinds:
        interaction = _create_interaction_sync(
            session=db_session,
            learner_id=1,
            item_id=1,
            kind=kind,
        )
        assert interaction.kind == kind


def test_create_interaction_with_very_long_kind(db_session: Session) -> None:
    """Test creating an interaction with a very long kind string."""
    long_kind = "a" * 1000
    interaction = _create_interaction_sync(
        session=db_session,
        learner_id=1,
        item_id=1,
        kind=long_kind,
    )
    assert interaction.kind == long_kind
    assert len(interaction.kind) == 1000


def test_interaction_log_create_model() -> None:
    """Test InteractionLogCreate model with various inputs."""
    log_create = InteractionLogCreate(learner_id=42, item_id=99, kind="attempt")
    assert log_create.learner_id == 42
    assert log_create.item_id == 99
    assert log_create.kind == "attempt"


def test_interaction_log_create_with_zero_ids() -> None:
    """Test InteractionLogCreate model with zero learner_id and item_id."""
    log_create = InteractionLogCreate(learner_id=0, item_id=0, kind="view")
    assert log_create.learner_id == 0
    assert log_create.item_id == 0


def test_interaction_model_response_schema() -> None:
    """Test InteractionModel response schema construction."""
    created_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    model = InteractionModel(
        id=1,
        learner_id=42,
        item_id=99,
        kind="attempt",
        created_at=created_at,
    )
    assert model.id == 1
    assert model.learner_id == 42
    assert model.item_id == 99
    assert model.kind == "attempt"
    assert model.created_at == created_at


def test_read_interactions_empty_database(db_session: Session) -> None:
    """Test reading interactions from an empty database."""
    interactions = _read_interactions_sync(db_session)
    assert interactions == []


def test_read_interactions_multiple_entries(db_session: Session) -> None:
    """Test reading multiple interactions from the database."""
    _create_interaction_sync(db_session, learner_id=1, item_id=1, kind="attempt")
    _create_interaction_sync(db_session, learner_id=2, item_id=2, kind="view")
    _create_interaction_sync(db_session, learner_id=1, item_id=3, kind="complete")

    interactions = _read_interactions_sync(db_session)
    assert len(interactions) == 3
    assert {i.kind for i in interactions} == {"attempt", "view", "complete"}
