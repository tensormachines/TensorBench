FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace
COPY benchmark.py /workspace/benchmark.py
RUN mkdir -p /workspace/results

ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "60", "--batch-sizes", "8", "4", "2", "1", "--grid-sizes", "2048", "1536", "1024", "--max-vram-gb", "12"]
