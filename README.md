Install from scratch
-----------
Need to build mxnet from source
	
Clone from: https://github.com/tornadomeet/mxnet/tree/face
	
Make sure you are on the mxnet/face branch using `git branch -a` and `git checkout`

	`sudo apt-get install -y libopenblas-dev liblapack-dev`

	`sudo apt-get install -y libopencv-dev`

	`cd mxnet`

	`make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas`

	`sudo apt-get install -y python-dev python-setuptools python-numpy python-pip`

	`cd python`

	`python setup.py build`

	`python setup.py install`
