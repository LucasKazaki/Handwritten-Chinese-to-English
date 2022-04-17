# Handwritten-Chinese-to-English
Taking an image of chinese text as an input and returning a translation using Python.

# SETUP
Before running, install deep_translator, opencv, Pillow, matplotlib, numpy, and, most importantly, Openvino. To install Openvino, follow this guide: https://pypi.org/project/openvino-dev/. 
Check to see if Openvino is compatable with your version of Python using the guide. Openvino takes a couple of minutes to install, so feel free to take a stretch break. Download this file and place in the same file as the Python file: https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/handwritten-simplified-chinese-recognition-0001/FP16/handwritten-simplified-chinese-recognition-0001.bin

# HOW TO USE
Download all files as well as the file above. If you have a supported version of Python, type ```python HCTE.py``` into your command prompt. If you do not have a supported version of Python, try ```py -3.9 HCTE.py```. 

# Known Issues
When the input image is bigger than the input for the model, the whole thing doesn't work. Resizing the image while maintaining the image has proved difficult. When testing, use images smaller than 96 x 2000 to circumvent the issue.
