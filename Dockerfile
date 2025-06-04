FROM        python:3.12-slim-bookworm AS base

COPY        --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

WORKDIR     /usr/src/app

ENV         PYTHONPATH=/usr/src/app \
            PATH="/usr/src/app/.venv/bin:$PATH"

RUN         apt-get update && apt-get install -y --no-install-recommends \
            && apt-get clean \
            && rm -rf /var/lib/apt/lists/* \
            && uv venv /usr/src/app/.venv

RUN         --mount=type=cache,target=/root/.cache/uv \
            --mount=type=bind,source=uv.lock,target=uv.lock \
            --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
            uv sync --locked --python=/usr/src/app/.venv/bin/python

COPY        ./src .

EXPOSE      9555

ENTRYPOINT  [ "python", "-m", "uvicorn", "app:app" ]
CMD         [ "--host=0.0.0.0", "--port=9555" ]
