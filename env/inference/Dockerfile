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
    software-properties-common \
    dirmngr \
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

COPY env/inference/pyproject.toml /app/
COPY env/inference/poetry.lock /app/
RUN poetry install -E extra_libraries -E gpu --with dev -vvv --no-root --no-directory

ENV PYTHONPATH=/app

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.gpg > /dev/null \
    && echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" | tee /etc/apt/sources.list.d/cran_r.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    cmake \
    pandoc \
    libwebp-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libuv1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RENV_PATHS_CACHE=/renv/cache

COPY renv.lock renv.lock
COPY renv/activate.R renv/activate.R
COPY renv/settings.json renv/settings.json

RUN sed -i 's|"https://cloud.r-project.org"|"https://packagemanager.posit.co/cran/__linux__/jammy/latest"|g' renv.lock \
    && sed -i 's|"https://cran.rstudio.com"|"https://packagemanager.posit.co/cran/__linux__/jammy/latest"|g' renv.lock \
    && sed -i 's|"https://cran.rstudio.com/"|"https://packagemanager.posit.co/cran/__linux__/jammy/latest"|g' renv.lock

RUN R -e "install.packages('renv', repos='https://cloud.r-project.org/', type='source')" \
    && R -e "renv::restore()"
