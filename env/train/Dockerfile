FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \    
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git \
    ca-certificates \
    emacs \
    tmux \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

ENV HOME="/root"
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$HOME/.local/bin:$PATH"

RUN curl https://pyenv.run | bash

ENV PYTHON_VERSION=3.9.7
RUN pyenv install $PYTHON_VERSION && pyenv global $PYTHON_VERSION

RUN which python 
RUN python --version

ENV POETRY_HOME="$HOME/.local" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0 || (cat /poetry-installer-error-*.log && exit 1)

WORKDIR /app

COPY ./env/train/pyproject.toml /app/
COPY ./env/train/poetry.lock /app/
RUN poetry install -E extra_libraries -E gpu --with dev -vvv --no-root --no-directory

ENV PYTHONPATH=/app
