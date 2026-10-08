FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace
COPY benchmark.py /workspace/benchmark.py
RUN mkdir -p /workspace/results

ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "60", "--batch-sizes", "64", "32", "16", "--image-size", "224", "--in-channels", "64", "--out-channels", "128", "--max-vram-gb", "12"]
