# Usa uma imagem oficial do PyTorch com suporte a CUDA
FROM pytorch/pytorch:latest

# Define o diretório de trabalho
WORKDIR /app

# Evita prompts interativos no apt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    nano \
    curl \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pacotes existentes (opcional)
RUN apt-get update && apt-get upgrade -y

# Define a versão do Poetry a ser instalada
ENV POETRY_VERSION=2.1.2

# Instala o Poetry na versão especificada
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 -

# Adiciona o Poetry ao PATH
ENV PATH="/root/.local/bin:$PATH"

# Desativa a criação de ambientes virtuais pelo Poetry para facilitar o uso no container
ENV POETRY_VIRTUALENVS_CREATE=false

# Define o comando padrão ao iniciar o container
CMD ["/bin/bash"]
