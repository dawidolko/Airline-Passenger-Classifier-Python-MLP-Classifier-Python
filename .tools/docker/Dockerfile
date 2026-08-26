# syntax=docker/dockerfile:1

# =============================================================================
# airline — reproducible experiment + dashboard image.
#
# The experiment is run at container start rather than baked into the image, so
# a rebuild is not needed to regenerate results after changing a parameter, and
# the generated artefacts stay on a volume the host can inspect.
# =============================================================================

FROM python:3.12-slim AS runtime

# Dependencies first: this layer is cached until requirements.txt changes, so an
# edit to the source does not reinstall the whole scientific stack.
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit writes a config and a metrics file into the home directory; running
# as a non-root user without one makes it fail on first launch.
RUN useradd --create-home --uid 1000 analyst \
    && chown -R analyst:analyst /app
USER analyst

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

# The pipeline is idempotent, so re-running it on every boot is safe and keeps
# the dashboard from starting against missing or stale artefacts.
CMD ["sh", "-c", "python run_experiment.py && exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
