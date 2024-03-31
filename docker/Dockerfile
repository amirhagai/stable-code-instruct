ARG MODELS_CACHE_DIR="/media/amir/HDD3/HF_CACHE"

# nvidia tags can be found at - https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:23.08-py3 

# Define Conda environment and path variables
ENV CONDA_DIR=/root/miniconda3 \
    PATH=${CONDA_DIR}/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1" \
    HF_HOME=${MODELS_CACHE_DIR} \
    TRANSFORMERS_CACHE=${MODELS_CACHE_DIR}

ARG DEBIAN_FRONTEND=noninteractive

ENV CONDA_DIR=/root/miniconda3 \
    PATH=${CONDA_DIR}/bin:${PATH}

# Install Miniconda and initialize conda in one RUN command to ensure the same shell session
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -u -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda init bash && \
    ${CONDA_DIR}/bin/conda init zsh

# Create a Conda environment and install the transformers package
RUN /bin/bash -c "source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda create -n HF python=3.10 -y && \
    conda activate HF && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y && \
    conda install -n HF conda-forge::transformers -y && \ 
    conda install conda-forge::gradio -y"










#ARG MODELS_CACHE_DIR="/media/amir/HDD3/HF_CACHE" 

#FROM nvcr.io/nvidia/pytorch:23.12-py3

#ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
#    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
#    FORCE_CUDA="1"
    
#RUN mkdir -p ~/miniconda3
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
#RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
#RUN rm -rf ~/miniconda3/miniconda.sh

#RUN ~/miniconda3/bin/conda init bash
#RUN ~/miniconda3/bin/conda init zsh

#RUN conda create HF python=3.10
#RUN conda activate HF
#RUN export HF_HOME=${MODELS_CACHE_DIR}
#RUN export TRANSFORMERS_CACHE=${MODELS_CACHE_DIR}
#RUN conda install conda-forge::transformers


