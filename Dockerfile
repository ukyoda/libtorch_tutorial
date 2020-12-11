from nvcr.io/nvidia/tensorrt:20.07-py3

RUN    apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y software-properties-common \
    && apt-get clean

RUN    bash /opt/tensorrt/install_opensource.sh \
    && bash /opt/tensorrt/python/python_setup.sh

# 依存パッケージインストール
RUN    apt-get -y install build-essential cmake checkinstall ccache \
                          libgtk-3-dev libjpeg-dev libpng++-dev \
    && apt-get -y install x264 libavformat-dev libavcodec-dev libswscale-dev libv4l-dev libatlas-base-dev libxvidcore-dev libx264-dev \
    && apt-get -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libfaac-dev \
    && apt-get -y install wget v4l-utils lsb-release lsb-core curl \
    && apt-get -y install git unzip pkg-config yasm zlib1g-dev libopenblas-dev liblmdb-dev \
    && apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler \
    && apt-get -y upgrade curl libssl-dev

# CMakeはせっかくだから最新版を入れよう
WORKDIR /usr/local/src
RUN    wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz \
    && tar xvzf cmake-3.19.1.tar.gz \
    && cd cmake-3.19.1 \
    && ./bootstrap --system-curl && make -j$(nproc) \
    && make install \
    && cd ../ \
    && rm -rf cmake.tar.gz

# glog, gflagsインストール
RUN    apt-get update \
    && apt-get -y install libgoogle-glog-dev libgoogle-glog-dev \
    && apt-get clean

# numpyとprotobufをインストール
RUN    pip3 install numpy protobuf

# OpenCVをインストール
WORKDIR /usr/local/src/opencv
RUN    wget https://github.com/opencv/opencv/archive/4.4.0.tar.gz -O opencv.tar.gz \
    && wget https://github.com/opencv/opencv_contrib/archive/4.4.0.tar.gz -O opencv-contrib.tar.gz \
    && mkdir opencv && tar xvzf opencv.tar.gz -C opencv --strip-components 1 \
    && mkdir opencv-contrib && tar xvzf opencv-contrib.tar.gz -C opencv-contrib --strip-components 1 \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_opencv_java=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv/opencv-contrib/modules \
        -D WITH_CUDA=ON \
        -D BUILD_TIFF=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D WITH_TBB=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D WITH_JPEG=ON \
        .. \
    && make -j$(nproc) \
    && make install \
    && cd /usr/local/src/opencv \
    && rm -f opencv.tar.gz \
    && rm -f opencv-contrib.tar.gz

RUN pip3 install torch==1.7.0 torchvision==0.8.1

WORKDIR /root
