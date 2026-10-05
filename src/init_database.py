"""Create the prediction-history database and apply schema migrations.

All credentials come from the environment (``DATABASE_URL`` or the standard
``PG*`` variables); nothing is hard-coded and no password is ever accepted on
the command line, printed, or logged.

Usage (Windows example; the local PostgreSQL 16 service listens on port 5433):

    set DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5433/skin_lesion_ai
    python src/init_database.py

The script connects to the server's maintenance database "postgres" using the
same credentials, creates the configured database if it does not exist, and
then opens the API's connection pool, which applies the schema migrations in
one transaction per migration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import psycopg.conninfo
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import PredictionStore, PredictionStoreError, describe_target, resolve_conninfo  # noqa: E402


def ensure_database(conninfo: str) -> str:
    """Create the configured database if it does not already exist."""
    settings = psycopg.conninfo.conninfo_to_dict(conninfo or "")
    dbname = settings.get("dbname") or os.getenv("PGDATABASE")
    if not dbname:
        raise PredictionStoreError(
            "No database name configured. Include the database name in DATABASE_URL "
            "or set PGDATABASE."
        )
    admin_conninfo = psycopg.conninfo.make_conninfo(conninfo, dbname="postgres")
    with psycopg.connect(admin_conninfo, autocommit=True) as admin_connection:
        exists = admin_connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if exists:
            print(f"Database {dbname!r} already exists.")
        else:
            # CREATE DATABASE cannot run inside a transaction; the connection
            # above uses autocommit. sql.Identifier quotes the name safely.
            admin_connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
            )
            print(f"Created database {dbname!r}.")
    return str(dbname)


def main() -> int:
    conninfo = resolve_conninfo()
    if conninfo is None:
        print(
            "No database configured. Set DATABASE_URL (or PGHOST/PGPORT/PGDATABASE/"
            "PGUSER/PGPASSWORD) first; credentials are never accepted as arguments.",
            file=sys.stderr,
        )
        return 2
    try:
        ensure_database(conninfo)
        store = PredictionStore.from_environment()
        if store is None:
            print("Unexpected state: database was configured but no store was created.", file=sys.stderr)
            return 1
        try:
            print("Applying schema migrations (create predictions table if missing)...")
            store.apply_migrations()
        finally:
            store.close()
    except PredictionStoreError as exc:
        print(f"Database initialization failed: {exc}", file=sys.stderr)
        return 1
    print(f"Prediction-history storage is ready ({describe_target(conninfo)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
