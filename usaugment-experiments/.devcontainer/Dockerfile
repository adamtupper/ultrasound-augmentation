FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG USERNAME=developer
ARG USER_UID=15266  # TODO: Replace with the user ID of the host user
ARG USER_GID=110  # TODO: Replace with the group ID of the host user

ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"

# Set timezone
ENV TZ=Canada/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.11 and non-Python dependencies
RUN apt-get clean && apt-get update && apt-get install -y software-properties-common net-tools
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get clean && apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils binfmt-support python3-pip git wget texlive-latex-extra

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME

# Create workspace directory
RUN mkdir -p /home/$USERNAME/workspace/

# Install virtualenv
RUN pip3 install virtualenv

# Set the working directory
WORKDIR /home/$USERNAME/workspace/