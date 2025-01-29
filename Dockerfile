FROM debian:bullseye-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    git \
    g++ \
    cmake \
    # for MKL
    apt-transport-https gpg-agent gnupg2 \
    # for kenlm
    libboost-thread-dev libboost-test-dev libboost-system-dev libboost-program-options-dev \
    # for arrayfire CPU backend
    libboost-stacktrace-dev \
    # OpenBLAS
    libopenblas-dev liblapacke-dev \
    # ATLAS
    libatlas3-base libatlas-base-dev liblapacke-dev \
    # FFTW
    libfftw3-dev \
    # ssh for OpenMPI
    openssh-server openssh-client \
    # for OpenMPI
    libopenmpi-dev openmpi-bin \
    # for kenlm
    zlib1g-dev libbz2-dev liblzma-dev \
  && apt-get clean \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/*
        
RUN cd /tmp \
  && git clone --branch v3.8.3 --depth 1 --recursive --shallow-submodules https://github.com/arrayfire/arrayfire.git \
  && mkdir -p arrayfire/build \
  && cd arrayfire/build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DAF_BUILD_CPU=ON \
              -DAF_BUILD_CUDA=OFF \
              -DAF_BUILD_OPENCL=OFF \
              -DAF_BUILD_EXAMPLES=OFF \
              -DAF_WITH_IMAGEIO=OFF \
              -DBUILD_TESTING=OFF \
              -DAF_BUILD_DOCS=OFF \
  && make install -j$(nproc) \
  && rm -rf /tmp/arrayfire
    
RUN cd /tmp \
  && git clone --branch v2.5.2 --depth 1 https://github.com/oneapi-src/onednn.git \
  && mkdir -p onednn/build \
  && cd onednn/build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DDNNL_BUILD_EXAMPLES=OFF \
  && make install -j$(nproc) \
  && rm -rf /tmp/onednn
    
RUN cd /tmp \
  && git clone https://github.com/facebookincubator/gloo.git \
  && cd gloo \
  && git checkout 56b221c0a811491d2dc2a3254b468ad687bbdaab \
  && mkdir build \
  && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DUSE_MPI=ON \
  && make install -j$(nproc) \
  && rm -rf /tmp/gloo
    
RUN cd /tmp \
  && git clone https://github.com/kpu/kenlm.git \
  && cd kenlm \
  && git checkout 9af679c38477b564c26917a5dcf52d2c86177fb9 \
  && mkdir build \
  && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  && make install -j$(nproc) \
  && rm -rf /tmp/kenlm
    
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    vim \
    nano \
    ffmpeg \
    # libsndfile
    libsndfile1-dev \
    # gflags
    libgflags-dev libgflags2.2 \
    # for glog
    libgoogle-glog-dev libgoogle-glog0v5 \
    # python sox
    sox libsox-dev python3-dev python3-pip python3-distutils \
  && apt-get clean \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install --upgrade setuptools numpy sox tqdm
        
RUN cd /tmp \
  && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
  && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
  && sh -c 'echo deb [trusted=yes] https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' \
  && apt-get update \
  && apt-get install -y --no-install-recommends intel-mkl-64bit-2020.4-912 \
  && rm -rf /tmp/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
  && apt-get clean \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
  && git clone https://github.com/flashlight/flashlight.git \
  && cd flashlight \
  && git checkout v0.3.1 \
  && mkdir -p build \
  && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DFL_BACKEND=CPU \
              -DFL_BUILD_ALL_APPS=OFF \
              -DFL_BUILD_PKG_TEXT=ON \
              -DFL_BUILD_PKG_RUNTIME=ON \
              -DFL_BUILD_PKG_SPEECH=ON \
              -DFL_BUILD_TESTS=OFF \
              -DFL_BUILD_EXAMPLES=OFF \
              -DFL_BUILD_APP_ASR=ON \
              -DFL_BUILD_APP_ASR_TOOLS=ON \
  && make install -j$(nproc)  \
  && cp /tmp/flashlight/cmake/FindCBLAS.cmake /usr/local/share/flashlight/cmake \
  && rm -rf /tmp/flashlight

COPY . /tmp/acoustic
RUN cd /tmp/acoustic \
  && python3 -m pip --no-cache-dir install . \
  && rm -rf speechmatching \
  && python3 init_models.py \
  && cd acoustic \
  && mkdir -p build \
  && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_MODULE_PATH=/usr/local/share/flashlight/cmake \
  && make \
  && mkdir /opt/bin \
  && cp acoustic /opt/bin \
  && rm -rf /tmp/acoustic

ENV ACOUSTIC_RUNNING_IN_DOCKER 1

WORKDIR /app

