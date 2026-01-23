# Multi-stage build for AI-Practices

FROM python:3.10-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base as dependencies

COPY requirements.txt pyproject.toml ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy pandas scipy matplotlib seaborn scikit-learn tqdm && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

FROM dependencies as development

RUN pip install pytest pytest-cov pytest-xdist black ruff mypy pre-commit jupyter notebook

COPY . .

CMD ["bash"]

FROM dependencies as production

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
COPY 13-distributed-training/ ./13-distributed-training/
COPY 14-agents-reasoning/ ./14-agents-reasoning/

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "--version"]

FROM dependencies as jupyter

RUN pip install jupyter notebook jupyterlab ipywidgets

COPY . .

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

FROM development as test

CMD ["pytest", "-v", "--tb=short", "07-reinforcement-learning/", "10-large-language-models/"]
