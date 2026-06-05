"""
Database initialization script.
Creates all tables in the PostgreSQL database.

Usage:
    python scripts/init_db.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database import create_tables


def main():
    print("[DB] Initializing database tables...")
    try:
        create_tables()
        print("[DB] All tables created successfully.")
    except Exception as exc:
        print(f"[DB] Error: {exc}")
        print("[DB] Make sure PostgreSQL is running and DATABASE_URL is correct in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
