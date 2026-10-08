FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace
COPY benchmark.py /workspace/benchmark.py
RUN mkdir -p /workspace/results

ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "60", "--working-set-gb", "1.5", "--index-fraction", "0.25", "--max-vram-gb", "12"]
