# FROM nvcr.io/nvidia/pytorch:24.05-py3
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV FORCE_CUDA="1"

# Obtain the UID and GID of the current user to create a user with the same ID, this is to avoid permission issues when mounting local volumes.
ARG USER
ARG USER_ID
ARG USER_GID

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add user.
RUN groupadd -g $USER_GID $USER \
    && useradd --uid $USER_ID --gid $USER_GID -m $USER \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

RUN apt install sudo libgl1-mesa-glx mesa-utils libglapi-mesa libqt5gui5 -y
RUN apt-get install -y build-essential cmake git curl ca-certificates \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    python3-tk \
    wget \
    jupyter \
    tmux

RUN apt-get update && apt-get install -y \
    libxrandr2 \
    libxrender1 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libfontconfig1 \
    libfreetype6 \
    x11-xserver-utils \
    fonts-dejavu

RUN apt-get install -y libglu1-mesa libgl1-mesa-glx libgl1-mesa-dri

RUN python -m pip install --upgrade pip
ENV PATH="${PATH}:/home/$USER/.local/bin"

RUN mkdir /.cache
COPY requirements.txt /.cache

USER $USER

RUN pip install -r /.cache/requirements.txt

# Disable jupyter authentication
WORKDIR /home/$USER
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py

# Change terminal color
ENV TERM=xterm-256color
RUN echo "PS1='\[\e[91m\]\u\[\e[0m\]@\[\e[93m\]\h\[\e[0m\]:\[\e[35m\]\w\[\e[0m\] > '" >> ~/.bashrc
RUN echo "source ~/.bashrc" >> ~/.bash_profile

WORKDIR /home/$USER/workspace
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]