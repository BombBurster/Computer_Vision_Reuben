opencv :- cv2
numpy
scipy
matplotlib
random

If issues with mpeg driver arise please follow the form:
https://askubuntu.com/questions/214421/how-to-install-the-mpeg-4-aac-decoder-and-the-h-264-decoder

If issues arise on Ubuntu with cv2.DestroyAllWindows() please follow the following instructions:
- conda remove opencv
- conda install -c menpo opencv
- pip install --upgrade pip
- pip install opencv-contrib-python