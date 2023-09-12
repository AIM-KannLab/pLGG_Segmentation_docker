FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
#anibali/pytorch:1.13.1-cuda11.7-ubuntu22.04
# docker pull anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y tzdata

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3-pip python3-setuptools python3-distutils && \
    apt-get install -y wget && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip 

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -n base -c defaults conda
COPY environment.yml /environment.yml 
RUN conda env create -f /environment.yml

## Create /entry.sh which will be our new shell entry point. This performs actions to configure the environment
## before starting a new shell (which inherits the env).
## The exec is important! This allows signals to pass
RUN     (echo '#!/bin/bash' \
    &&   echo '__conda_setup="$(/opt/conda/bin/conda shell.bash hook 2> /dev/null)"' \
    &&   echo 'eval "$__conda_setup"' \
    &&   echo 'conda activate "${CONDA_TARGET_ENV:-base}"' \
    &&   echo '>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"' \
    &&   echo 'exec "$@"'\
        ) >> /entry.sh && chmod +x /entry.sh

SHELL ["/entry.sh", "/bin/bash", "-c"]
ENV CONDA_TARGET_ENV=bwh_plgg
RUN conda init && echo 'conda activate "${CONDA_TARGET_ENV:-base}"' >>  ~/.bashrc
#ENTRYPOINT ["/entry.sh"]

COPY nnUnet/ /nnUnet/
COPY run.py /run.py
COPY HDBET_Code/ /HDBET_Code/
COPY example_input/ /example_input/ 
COPY temp_dir/ /temp_dir/
COPY reload_bash /reload_bash
RUN chmod +x /reload_bash
ENV RESULTS_FOLDER=/nnUnet/nnUnet_trained_models/

WORKDIR /nnUnet
RUN ["conda", "run", "-n", "bwh_plgg", "pip", "install", "-e", "."]

WORKDIR /
ENV MKL_SERVICE_FORCE_INTEL=1
ENTRYPOINT ["conda", "run", "-n", "bwh_plgg", "python", "/run.py"]




