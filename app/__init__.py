"""Application package for ChronoML's HTTP interface. It holds the FastAPI
entrypoint, request and response schemas, and route handlers that orchestrate
inference, logging, and replay. Keeping this package separate from model
training and database code enforces a clean API boundary and makes deployment
straightforward. When you change API behavior, you can do so here without
touching model or storage internals directly."""
