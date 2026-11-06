# Prefixwise proxy. Standard library only, so the image is just Python.
#   docker build -t prefixwise .
#   docker run -p 127.0.0.1:8080:8080 -v prefixwise-data:/data prefixwise               # mock mode
#   docker run -p 127.0.0.1:8080:8080 -v prefixwise-data:/data \
#     -e PREFIXWISE_UPSTREAM_URL=https://api.openai.com prefixwise                       # real upstream
# Publish on 127.0.0.1 as above. The container listens on 0.0.0.0, so with
# PREFIXWISE_API_KEY set it refuses to start unless PREFIXWISE_AUTH_TOKEN is
# set too (clients then send it as X-Prefixwise-Token).
# No pip install: there are no runtime dependencies, so the image builds offline.
FROM python:3.12-slim
WORKDIR /app
COPY src ./src
RUN useradd --create-home --uid 10001 prefixwise && mkdir /data && chown prefixwise /data
USER prefixwise
ENV PYTHONPATH=/app/src PREFIXWISE_LEDGER_PATH=/data/prefixwise.db PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"
ENTRYPOINT ["python", "-m", "prefixwise"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
