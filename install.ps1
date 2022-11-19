pip install diffusers
pip install transformers
pip install accelerate

# I'm using python 3.10 and super-image has dependency issues with opencv-python (selecting specific versions)
pip install super-image --no-deps
pip install h5py
pip install opencv-python

pip install fastapi uvicorn python-multipart

# I've encountered issues installing leveldb on windows, if you are using unix, can replace with leveldb
pip install plyvel-wheels