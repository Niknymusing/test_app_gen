FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    wget \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user for security
RUN groupadd -r meuuser && useradd -r -g meuuser -d /app -s /bin/bash meuuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN chown -R meuuser:meuuser /usr/local/lib/python3.11/site-packages

# Copy source code, data, and entrypoint
COPY src/ ./src/
COPY data/ ./data/
COPY entrypoint.sh /app/entrypoint.sh

# Create workspace directory and set permissions (including entrypoint)
RUN mkdir -p /workspace/logs \
    && chmod +x /app/entrypoint.sh \
    && chown -R meuuser:meuuser /workspace \
    && chown -R meuuser:meuuser /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV WORKSPACE_PATH=/workspace
ENV HOST=0.0.0.0
ENV PORT=8000

# Switch to non-root user
USER meuuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with entrypoint to configure Git
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "src/main.py"]