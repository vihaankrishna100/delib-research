from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = "sqlite:///./deliberation.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations():
    """
    Lightweight, idempotent SQLite migration.

    `Base.metadata.create_all` creates NEW tables but never ALTERs existing ones,
    so columns added to already-existing tables must be patched in by hand. This
    adds any missing columns with `ALTER TABLE ... ADD COLUMN` and is safe to run
    on every startup (it checks PRAGMA table_info first and is a no-op if present).
    """
    from sqlalchemy import text

    # (table, column, column-definition)
    additions = [
        ("deliberation_posts", "is_prelude", "BOOLEAN DEFAULT 0"),
        ("events",             "is_prelude", "BOOLEAN DEFAULT 0"),
        ("recommendation_logs", "is_prelude", "BOOLEAN DEFAULT 0"),
        ("recommendation_logs", "skip_reason", "TEXT"),
        ("recommendation_logs", "confidence", "REAL"),
        ("persona_topic_headers", "confidence", "REAL DEFAULT 0.5"),
        ("persona_topic_headers", "parent_id", "TEXT"),
        ("persona_topic_headers", "action_scores", "JSON"),
        ("user_model_snapshots", "tree_json", "JSON"),
    ]

    with engine.begin() as conn:
        for table, column, coldef in additions:
            # Skip silently if the table doesn't exist yet (create_all handles it).
            try:
                cols = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            except Exception:
                continue
            if not cols:
                continue
            if column not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}"))
