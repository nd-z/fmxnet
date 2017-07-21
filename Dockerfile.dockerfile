FROM python:2.7-slim

WORKDIR /fmxnet

#TODO for the container, replace mxnet-face with mxnet b/c need to build from source
ADD . /fmxnet

#necessary for dependencies
EXPOSE 80

RUN pip install --upgrade -r requirements.txt

#install OpenCV and dependencies for mxnet
RUN apt-get update && apt-get install -y \
	python-dev
	python-setuptools
	python-pip
	libopencv-dev python-opencv \
	make \
	cmake \
	libopenblas-dev \
	liblapack-dev \

#build mxnet from source
RUN cd mxnet
RUN -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
RUN cd python

ENV PYTHONPATH /mxnet/python

RUN python setup.py build
RUN python setup.py install
RUN cd ../../

#run the detection script
CMD ["python", "detect.py"]