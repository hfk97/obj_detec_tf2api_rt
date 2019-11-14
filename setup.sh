pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install cython

git clone https://github.com/tensorflow/models.git

cd models/research

protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


pip install .
