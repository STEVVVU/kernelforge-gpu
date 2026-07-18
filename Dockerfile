FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build

ENTRYPOINT ["./build/kernelforge_bench"]

