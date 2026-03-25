FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for caching
COPY pyproject.toml .
COPY vendor/tinker-cookbook/pyproject.toml vendor/tinker-cookbook/pyproject.toml
COPY vendor/tinker-cookbook/tinker_cookbook vendor/tinker-cookbook/tinker_cookbook

# Install dependencies (CPU-only torch to keep image smaller)
RUN uv venv && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install -e vendor/tinker-cookbook && \
    uv pip install tinker>=0.14.0

COPY serve.py .

EXPOSE 8787

CMD ["/app/.venv/bin/python", "serve.py", "--port", "8787"]
