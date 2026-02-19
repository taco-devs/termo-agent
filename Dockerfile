FROM python:3.12-slim

# System deps for nanobot (git for tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install nanobot from PyPI (vanilla, unmodified)
RUN uv pip install --system --no-cache nanobot-ai

# Install termo-agent from source (not on PyPI yet)
COPY pyproject.toml /tmp/termo-agent/
COPY termo_agent/ /tmp/termo-agent/termo_agent/
RUN uv pip install --system --no-cache /tmp/termo-agent && rm -rf /tmp/termo-agent

# Config volume
VOLUME /root/.nanobot

WORKDIR /root/.nanobot
EXPOSE 3015

ENTRYPOINT ["termo-agent"]
CMD ["--adapter", "nanobot"]
