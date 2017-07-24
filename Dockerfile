FROM python:2.7-slim

WORKDIR /fmxnet
RUN pwd
RUN ls

#TODO for the container, replace mxnet-face with mxnet b/c need to build from source
ADD . /fmxnet

#necessary for dependencies
EXPOSE 80

#install OpenCV and dependencies for mxnet
RUN apt-get update && apt-get install -y \
    python-setuptools \
    python-pip \
    libopencv-dev python-opencv \
    make \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgtk-3-dev \
    libboost-all-dev

RUN pip install --upgrade -r requirements.txt

#build mxnet from source
WORKDIR "mxnet"
RUN pwd
RUN ls
RUN make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas

WORKDIR "python"

#TODO how do I add this to the pythonpath?
#ENV PYTHONPATH /mxnet/python

RUN python setup.py build
RUN python setup.py install
#RUN cd ../../

#run the detection script
CMD ["python", "./facenet/src/align/detect.py"]
