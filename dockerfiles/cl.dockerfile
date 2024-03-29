# sudo docker build --force-rm -f dockerfiles/cl.dockerfile -t cl:1.0 .
# sudo apt-get install x11-xserver-utils
# xhost +
# sudo docker run --gpus all --cpuset-cpus=0-39 -m 250g --shm-size=250g -it -v /home/hs/ML/CL_project:/home/CL_project --name cl cl:1.0

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $>TZ > /etc/timezone

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update 

RUN echo "== Install Basic Tools ==" &&\
    apt install -y --allow-unauthenticated \
    openssh-server vim nano htop tmux sudo git unzip build-essential iputils-ping net-tools ufw \
    python3 python3-pip curl dpkg libgtk2.0-dev \
    cmake libwebp-dev ca-certificates gnupg git \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran \
    libgl1-mesa-glx libglu1-mesa-dev x11-utils x11-apps quadprog


# ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
#     echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
#     ldconfig

RUN echo "== Install Dev Tolls ==" &&\
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 &&\
    pip3 install opencv-python matplotlib pillow torchsummary tqdm scipy

# githup
RUN cd /home/ &&\
    git clone https://github.com/kongbuhaja/CL_project.git &&\
    git config --global --add safe.directory /home/CL_project