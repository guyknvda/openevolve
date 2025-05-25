#!/bin/bash
# DIR is the directory where the script is saved (should be <project_root/scripts)
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $DIR

MY_UID=$(id -u)
MY_GID=$(id -g)
MY_UNAME=$(id -un)
# Note : based on the driver installed (see nvidia-smi) find the supported cuda version. 

# BASE_IMAGE=nvcr.io/nvidia/nvhpc:23.9-devel-cuda12.2-ubuntu22.04

# cuda:12.3.1-devel-ubuntu22.04 is the one that aligns with cuda-samples 12.3 and builds on 12.2
# BASE_IMAGE=nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

# 25.01 : CUDA_DRIVER_VERSION=570.86.10 , CUDA_VERSION=12.8.0.038
BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3

# mkdir -p ${DIR}/.vscode-server
mkdir -p ${DIR}/.cursor-server

LINK=$(realpath --relative-to="/home/${MY_UNAME}" "$DIR" -s)
IMAGE=openevolve_12_8:latest
if [ -z "$(docker images -q ${IMAGE})" ]; then
    # Create dev.dockerfile
    FILE=dev.dockerfile

    ### Pick Tensorflow / Torch based base image below
    # echo "FROM nvcr.io/nvidia/tensorflow:23.01-tf2-py3" > $FILE
    echo "FROM $BASE_IMAGE" > $FILE

    echo "  RUN apt-get update" >> $FILE
    echo "  RUN apt-get -y install nano gdb time" >> $FILE
    echo "  RUN apt-get -y install nvidia-cuda-gdb" >> $FILE
    echo "  RUN apt-get -y install sudo" >> $FILE
    # echo "  RUN apt-get -y install build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libfreeimage-dev && rm -rf /var/lib/apt/lists/*" >> $FILE

    echo "  RUN (groupadd -g $MY_GID $MY_UNAME || true) && useradd --uid $MY_UID -g $MY_GID --no-log-init --create-home $MY_UNAME && (echo \"${MY_UNAME}:password\" | chpasswd) && (echo \"${MY_UNAME} ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers)" >> $FILE

    echo "  RUN mkdir -p $DIR" >> $FILE
    # echo "  RUN ln -s ${LINK}/.vscode-server /home/${MY_UNAME}/.vscode-server" >> $FILE
    echo "  RUN ln -s ${LINK}/.cursor-server /home/${MY_UNAME}/.cursor-server" >> $FILE
    echo "  RUN echo \"fs.inotify.max_user_watches=524288\" >> /etc/sysctl.conf" >> $FILE
    echo "  RUN sysctl -p" >> $FILE

    # echo "  WORKDIR /usr/local/cuda" >> $FILE
    # echo "  RUN git clone --branch v12.8 --depth 1 https://github.com/NVIDIA/cuda-samples.git" >> $FILE
    # echo "  RUN chown -R ${MY_UNAME}:${MY_GID} /usr/local/cuda/cuda-samples" >> $FILE
    # echo "  WORKDIR ${CUDA_HOME}/cuda-samples" >> $FILE
    # echo "  RUN make TARGET_ARCH=x86_64 SMS='89'" >> $FILE

    echo "  USER $MY_UNAME" >> $FILE
    echo "  COPY docker.bashrc /home/${MY_UNAME}/.bashrc" >> $FILE 
    
   # START: install any additional package required for your image here
    # echo "  COPY requirements.txt /home/${MY_UNAME}/requirements.txt" >> $FILE
    # echo "  RUN pip install -r /home/${MY_UNAME}/requirements.txt" >> $FILE
    echo "  RUN pip install transformers accelerate bitsandbytes peft datasets dotenv openai huggingface_hub[hf_xet]" >> $FILE
    # END: install any additional package required for your image here
    echo "  RUN . /home/${MY_UNAME}/.bashrc" >> $FILE
    echo "  WORKDIR $DIR/.." >> $FILE
    # echo "  RUN pip install -e $DIR/.." >> $FILE
    echo "  CMD /bin/bash" >> $FILE

    docker buildx build -f dev.dockerfile -t ${IMAGE} .
fi

EXTRA_MOUNTS=""
# HF_HOME=${HF_HOME:-/.cache/huggingface}
# if [ -d "/home/${MY_UNAME}/.cache/" ]; then
#     EXTRA_MOUNTS+=" --mount type=bind,source=/home/${MY_UNAME}/scratch,target=/home/${MY_UNAME}/scratch"
# fi
CACHE_FOLDER_ON_HOST=/home/${MY_UNAME}/scratch/.cache/
MOUNT_CACHE_FOLDER=" --mount type=bind,source=${CACHE_FOLDER_ON_HOST},target=/home/${MY_UNAME}/.cache"

CODE_FOLDER=/home/${MY_UNAME}/code
MOUNT_CODE_FOLDER="--mount type=bind,source=${CODE_FOLDER},target=${CODE_FOLDER}"

DATA_FOLDER=/home/${MY_UNAME}/data
MOUNT_DATA_FOLDER=" --mount type=bind,source=${DATA_FOLDER},target=${DATA_FOLDER}"

# Ensure the llmnet network exists
docker network inspect llmnet >/dev/null 2>&1 || docker network create llmnet

MODEL_NAME=qwen3:8b
docker run -d --rm --gpus all --name ollama --network llmnet -p 11434:11434 ollama/ollama

# Wait until Ollama API is ready
until curl -s http://localhost:11434 | grep -q 'Ollama'; do
    echo "Waiting for Ollama to be ready..."
    sleep 1
done

docker exec -it ollama ollama pull ${MODEL_NAME}

docker run \
    --gpus \"device=all\" \
    --privileged \
    --network llmnet \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
    --mount type=bind,source=${DIR}/..,target=${DIR}/.. \
    ${MOUNT_CODE_FOLDER} \
    ${MOUNT_CACHE_FOLDER} \
    ${MOUNT_DATA_FOLDER} \
    --shm-size=8g \
    ${IMAGE}

# -v /var/run/docker.sock:/var/run/docker.sock \
# ${MOUNT_CODE_FOLDER} \
# ${MOUNT_DATA_FOLDER} \
# --name dlr  \
# -p 8888:8888 -p 6006:6006 \
# --mount type=bind,source=${DIR}/..,target=${DIR}/.. \
 
cd -


