FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MKL_THREADING_LAYER=GNU
ENV PYTHONUNBUFFERED=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install exactly as README says
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install vllm==0.6.3
RUN pip install ray
RUN pip install flash-attn --no-build-isolation

# Install local verl with ToolRL reward functions
COPY . /workspace
RUN pip install -e /workspace && pip install mlflow

CMD ["sleep", "infinity"]
