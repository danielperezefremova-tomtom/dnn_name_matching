FROM databricksruntime/dbfsfuse:12.2-LTS 
# Oficial databricks docker runtimes

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

EXPOSE 8000

RUN sudo apt update && sudo apt-get install -y git 
RUN sudo apt install -y python3.9-dev pkg-config libcairo2-dev # dependencies originated in keras stuff
RUN git config --global --add safe.directory /workspace

COPY src/requirements.in /tmp/requirements.in
RUN pip install pip-tools 
RUN pip-compile --resolver legacy /tmp/requirements.in > /tmp/requirements.txt
RUN pip install -r /tmp/requirements.in
RUN pip install -U ipykernel
RUN pip install black

RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash # install az cli

USER $USERNAME
