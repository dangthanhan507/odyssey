# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:2.3.3 as micromamba

FROM ubuntu:24.04
LABEL Author An

###################################################################
# This Dockerfile sets up the development environment for Odyssey.
# 
#

# allows the use of bash in Dockerfile

WORKDIR /root/amazon_ws/

RUN apt-get upgrade && apt-get update

RUN apt-get install -y --no-install-recommends \
        git \
        vim \
        nano \
        tar \
        wget \
        zip \
        bzip2 \
        unzip \
        less \
        curl \
        tree \
        htop \
        net-tools \
        software-properties-common \
        apt-utils \
        apt-transport-https \
        python3-dev \
        python3-pip \
        python3-tk \
        default-jdk \
        libjchart2d-java \
        libjide-oss-java \
        libxmlgraphics-commons-java \
        liblcm-dev \
        pkg-config \
        libhidapi-dev \
    &&\
    rm -rf /var/lib/apt/lists/*


ARG MAMBA_USER=user
ARG MAMBA_USER_ID=57439
ARG MAMBA_USER_GID=57439
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"


COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh
USER $MAMBA_USER

USER root

ARG MAMBA_DOCKERFILE_ACTIVATE=1

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["/bin/bash"]
RUN micromamba install --yes --name base --channel conda-forge && \
    micromamba clean --all --yes
RUN micromamba create -y -n isaac python=3.12

# NOTE: can't run micromamba activate, so we use micromamba run instead
RUN micromamba run -n isaac python3 -m pip install uv

RUN echo "micromamba activate isaac" >> /root/.bashrc

# setup odyssey lcm for lcm-spy (message debugging)
RUN echo "export CLASSPATH=$CLASSPATH:/usr/share/java/lcm.jar" >> /root/.bashrc
RUN echo "cd /root/amazon_ws/odyssey/odyssey/msgs && source buildjar.sh && cd ~" >> /root/.bashrc


SHELL ["/bin/bash", "-c", "-l"]
# setup workspace

# RUN python3 -m pip install scipy numpy pyyaml matplotlib opencv-python