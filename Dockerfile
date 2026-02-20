FROM python:3.12-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster pip installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install termo-agent from source
COPY pyproject.toml /tmp/termo-agent/
COPY termo_agent/ /tmp/termo-agent/termo_agent/
RUN uv pip install --system --no-cache /tmp/termo-agent && rm -rf /tmp/termo-agent

# Data volume
VOLUME /home/sprite/agent

WORKDIR /home/sprite/agent
EXPOSE 8080

ENTRYPOINT ["termo-agent"]
CMD ["--adapter", "openai_agents"]
