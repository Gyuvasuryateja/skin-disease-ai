"""PostgreSQL persistence layer for prediction history.

Connection credentials come exclusively from the environment: either a single
``DATABASE_URL`` (``postgresql://user:password@host:port/dbname``) or the
standard libpq ``PG*`` variables (PGHOST, PGPORT, PGDATABASE, PGUSER,
PGPASSWORD). No secret is ever hard-coded or written to the logs.

The stored record deliberately avoids sensitive personal information: only the
sanitized client-side image filename is kept (as a display reference); the
uploaded image itself is never persisted.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
from typing import Any, Iterator

import psycopg
import psycopg.conninfo
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

try:
    from dotenv import load_dotenv

    # Load the project-local .env (git-ignored) once at import so every entry
    # point -- the API server, init_database.py, and the database tests -- sees
    # the same credentials. Values already present in the real environment win.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv is optional; real env vars still work.
    pass


LOGGER = logging.getLogger("skin_lesion_api.db")

# Session-level advisory lock so concurrent API processes cannot apply
# migrations at the same time. Arbitrary but stable application constant.
MIGRATION_LOCK_ID = 8_201_601_001
MAX_FILENAME_LENGTH = 200
PREDICTION_COLUMNS = (
    "id, image_filename, predicted_class, predicted_class_name, "
    "confidence, class_probabilities, created_at"
)

# Each migration is (version, name, statements) and runs exactly once, inside
# one transaction, recorded in schema_migrations.
MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (
        1,
        "create_predictions_table",
        (
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                image_filename text NOT NULL,
                predicted_class text NOT NULL,
                predicted_class_name text NOT NULL,
                confidence double precision NOT NULL
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
                class_probabilities jsonb NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS predictions_created_at_desc_idx
                ON predictions (created_at DESC, id DESC)
            """,
        ),
    ),
)

SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version integer PRIMARY KEY,
    name text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


class PredictionStoreError(RuntimeError):
    """Raised when the prediction-history database cannot be reached or updated."""


def sanitize_image_filename(filename: str | None) -> str:
    """Return a safe, path-free display reference for an uploaded filename.

    Upload filenames are client-supplied and untrusted: strip any path
    components, drop non-printable characters, and cap the length.
    """
    if not filename:
        return "unnamed-image"
    base = re.split(r"[\\/]", filename)[-1]
    base = "".join(character if character.isprintable() else "_" for character in base).strip()
    base = base[:MAX_FILENAME_LENGTH].strip()
    return base or "unnamed-image"


def resolve_conninfo() -> str | None:
    """Return connection configuration from the environment, or None if unset."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url
    if any(os.getenv(name, "").strip() for name in ("PGHOST", "PGDATABASE", "PGUSER")):
        # An empty conninfo lets libpq read the standard PG* variables.
        return ""
    return None


def describe_target(conninfo: str) -> str:
    """Describe the connection target without ever exposing the password."""
    try:
        settings = psycopg.conninfo.conninfo_to_dict(conninfo or "")
    except psycopg.Error:
        return "<unparseable connection string>"
    host = settings.get("host") or os.getenv("PGHOST") or "localhost"
    port = settings.get("port") or os.getenv("PGPORT") or "5432"
    dbname = settings.get("dbname") or os.getenv("PGDATABASE") or "<default database>"
    user = settings.get("user") or os.getenv("PGUSER") or "<operating-system user>"
    return f"database {dbname!r} at {host}:{port} as user {user!r}"


class PredictionStore:
    """Small, synchronous PostgreSQL store for completed predictions."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    @classmethod
    def from_environment(
        cls, min_size: int = 1, max_size: int = 4
    ) -> "PredictionStore | None":
        """Build a migrated store from environment configuration.

        Returns None when no database is configured. Raises
        PredictionStoreError when configuration exists but the server is
        unreachable or migrations fail; error messages never include secrets.
        """
        conninfo = resolve_conninfo()
        if conninfo is None:
            return None
        try:
            pool = ConnectionPool(
                conninfo=conninfo,
                min_size=min_size,
                max_size=max_size,
                timeout=30,
                max_idle=300,
                open=True,
                kwargs={"row_factory": dict_row, "autocommit": True},
                name="skin-lesion-history",
            )
        except Exception as exc:
            raise PredictionStoreError(
                f"Could not open the PostgreSQL connection pool for {describe_target(conninfo)}: {exc}"
            ) from exc
        store = cls(pool)
        try:
            store.apply_migrations()
        except Exception:
            store.close()
            raise
        return store

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        """Yield a pooled connection, mapping driver errors to a safe type."""
        try:
            with self._pool.connection() as connection:
                yield connection
        except psycopg.Error as exc:
            raise PredictionStoreError(f"Database operation failed: {exc}") from exc

    def apply_migrations(self) -> list[int]:
        """Apply pending schema migrations exactly once (safe to re-run)."""
        applied: list[int] = []
        with self._connection() as connection:
            connection.execute(SCHEMA_MIGRATIONS_TABLE_SQL)
            connection.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
            try:
                existing = {
                    int(row["version"])
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                }
                for version, name, statements in MIGRATIONS:
                    if version in existing:
                        continue
                    # One transaction per migration: a failed statement rolls
                    # back only that migration, and schema_migrations stays
                    # consistent with the tables on disk.
                    with connection.transaction():
                        for statement in statements:
                            connection.execute(statement)
                        connection.execute(
                            "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                            (version, name),
                        )
                    applied.append(version)
                    LOGGER.info("Applied database migration %d (%s).", version, name)
            finally:
                try:
                    connection.execute(
                        "SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,)
                    )
                except psycopg.Error:
                    LOGGER.warning(
                        "Could not release the migration advisory lock; PostgreSQL "
                        "will release it when the session closes."
                    )
        return applied

    def insert_prediction(
        self,
        *,
        image_filename: str,
        predicted_class: str,
        predicted_class_name: str,
        confidence: float,
        class_probabilities: dict[str, float],
    ) -> dict[str, Any]:
        """Store one completed prediction and return the stored row."""
        row = None
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO predictions
                    (image_filename, predicted_class, predicted_class_name,
                     confidence, class_probabilities)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, image_filename, predicted_class, predicted_class_name,
                          confidence, class_probabilities, created_at
                """,
                (
                    image_filename,
                    predicted_class,
                    predicted_class_name,
                    float(confidence),
                    Jsonb(class_probabilities),
                ),
            ).fetchone()
        if row is None:
            raise PredictionStoreError("Inserting the prediction returned no row.")
        return dict(row)

    def list_predictions(self, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """Return one newest-first page of predictions plus the total count."""
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset must be >= 0.")
        with self._connection() as connection:
            total = int(
                connection.execute("SELECT count(*) AS total FROM predictions").fetchone()[
                    "total"
                ]
            )
            rows = connection.execute(
                f"""
                SELECT {PREDICTION_COLUMNS}
                FROM predictions
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], total

    def get_prediction(self, prediction_id: int) -> dict[str, Any] | None:
        """Return one stored prediction row, or None when the id is unknown."""
        row = None
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {PREDICTION_COLUMNS} FROM predictions WHERE id = %s",
                (prediction_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_all_predictions(self) -> int:
        """Delete every stored prediction record and return how many were removed."""
        with self._connection() as connection:
            row = connection.execute(
                """
                WITH deleted AS (DELETE FROM predictions RETURNING 1)
                SELECT count(*) AS deleted_count FROM deleted
                """
            ).fetchone()
        return int(row["deleted_count"])

    def close(self) -> None:
        """Close the underlying connection pool."""
        try:
            self._pool.close()
        except Exception:
            LOGGER.exception("Error while closing the prediction-history connection pool")
