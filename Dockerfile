# AI-Practices Docker Image
# Multi-stage build for optimized image size

# ============== Base Stage ==============
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============== Dependencies Stage ==============
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ============== Development Stage ==============
FROM dependencies as development

# Install dev dependencies
RUN pip install pytest pytest-cov pytest-xdist black ruff mypy pre-commit jupyter notebook

# Copy source code
COPY . .

# Set default command
CMD ["bash"]

# ============== Production Stage ==============
FROM dependencies as production

# Copy only necessary files
COPY 01-foundations/ ./01-foundations/
COPY 02-neural-networks/ ./02-neural-networks/
COPY 03-computer-vision/ ./03-computer-vision/
COPY 04-sequence-models/ ./04-sequence-models/
COPY 05-advanced-topics/ ./05-advanced-topics/
COPY 06-generative-models/ ./06-generative-models/
COPY 07-reinforcement-learning/ ./07-reinforcement-learning/
COPY 08-theory-notes/ ./08-theory-notes/
COPY 09-practical-projects/ ./09-practical-projects/
COPY 10-large-language-models/ ./10-large-language-models/
COPY 11-multimodal-learning/ ./11-multimodal-learning/
COPY 12-deployment-optimization/ ./12-deployment-optimization/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "--version"]

# ============== Jupyter Stage ==============
FROM dependencies as jupyter

# Install Jupyter and extensions
RUN pip install jupyter notebook jupyterlab ipywidgets

# Copy source code
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ============== Test Stage ==============
FROM development as test

# Run tests
CMD ["pytest", "-v", "--tb=short", "07-reinforcement-learning/", "10-large-language-models/"]
