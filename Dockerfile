FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Avoid interactive prompts and set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 1. Install system packages and fonts
# Automatically accept the EULA for ttf-mscorefonts-installer to prevent build halts
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | \
    debconf-set-selections && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    git \
    cmake \
    build-essential \
    fontconfig \
    ttf-mscorefonts-installer \
    mesa-common-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update OS font cache
RUN fc-cache -fv

# Install Micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

RUN mkdir -p $MAMBA_ROOT_PREFIX/bin && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C $MAMBA_ROOT_PREFIX/bin --strip-components=1 bin/micromamba

# 2. Install Python libraries and Jupyter
RUN micromamba install -y -n base -c conda-forge \
    python=3.10 \
    puma \
    jupyterlab \
    ipympl \
    ipyvtklink \
    numpy \
    scipy \
    tifffile \
    tqdm \
    matplotlib \
    pandas \
    pyvista \
    numba \
    && micromamba clean --all --yes

# Physically remove Matplotlib font cache and safely rebuild it using the standard method
RUN rm -rf /root/.cache/matplotlib && \
    micromamba run -n base python -c "import matplotlib.font_manager; matplotlib.font_manager.FontManager()"

# 3. Clone and compile chfem_gpu
RUN git clone https://github.com/hikuram/chfem.git /opt/chfem && \
    cd /opt/chfem && \
    mkdir -p build && \
    cd build && \
    export NUMPY_INCLUDE=$(micromamba run -n base python -c "import numpy; print(numpy.get_include())") && \
    export C_INCLUDE_PATH=${NUMPY_INCLUDE}:${C_INCLUDE_PATH} && \
    export CPLUS_INCLUDE_PATH=${NUMPY_INCLUDE}:${CPLUS_INCLUDE_PATH} && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_C_FLAGS="-DCUDAPCG_MATKEY_32BIT" \
      -DCMAKE_CUDA_FLAGS="-DCUDAPCG_MATKEY_32BIT" && \
    make -j4

# Add compiled chfem executable to PATH
ENV PATH="/opt/chfem/:${PATH}"

# 4. Setup the PuMA tutorials
WORKDIR /workspace
RUN git clone https://github.com/nasa/puma.git

# 5. Setup working environment
WORKDIR /workspace
EXPOSE 8888

## Configure Jupyter Lab to start without a password
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
