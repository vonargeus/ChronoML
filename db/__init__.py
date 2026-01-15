"""Database package for ChronoML persistence. It provides schema creation and
retention cleanup helpers that the API uses to store prediction events
reliably. Centralizing database utilities here keeps SQL code consistent and
prevents duplication across the app and tests. The goal is to make event
storage predictable and easy to evolve while preserving replay and audit
requirements for long term use."""
