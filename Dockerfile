FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY packages/core/src ./packages/core/src
COPY apps/api/src ./apps/api/src
RUN pip install --no-cache-dir .
ENV PYTHONPATH=/app/packages/core/src:/app/apps/api/src
EXPOSE 8000
CMD ["uvicorn", "quant_os_api.main:app", "--app-dir", "apps/api/src", "--host", "0.0.0.0", "--port", "8000"]
