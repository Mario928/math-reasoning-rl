FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /workspace

# Install vllm, ray, flash-attn exactly as README says
RUN pip install vllm==0.6.3 && \
    pip install ray && \
    pip install flash-attn --no-build-isolation

# Install local verl (with ToolRL reward functions) at build time
COPY . /workspace
RUN pip install -e /workspace

CMD ["sleep", "infinity"]
