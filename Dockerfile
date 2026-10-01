# Petrophys MCP server over HTTP.
#
#   docker build -t petrophys-mcp .
#   docker run -p 8000:8000 -e PETROPHYS_MCP_TOKEN=<secret> -v "$PWD/wells:/data" petrophys-mcp
#
# Clients connect to http://<host>:8000/mcp with "Authorization: Bearer <secret>".
# Put your LAS/DLIS/CSV files in the mounted folder; exports are written there too.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_DATA.md MANIFEST.in ./
COPY src ./src
RUN pip install ".[mcp]" \
    && useradd --create-home --uid 1000 petrophys \
    && mkdir -p /data \
    && chown petrophys /data

USER petrophys
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

# The token comes from PETROPHYS_MCP_TOKEN; the server refuses to start without one.
ENTRYPOINT ["petrophys-mcp", "serve", "--http", "--host", "0.0.0.0", "--port", "8000", "--root", "/data"]
