import os
'''os.system("pip install deep_translator")
os.system("pip install Pillow")
os.system("pip install opencv-python")
os.system("pip install matplotlib")
os.system("pip install numpy")
os.system("py -3.9 -m pip install --user virtualenv")
os.system("py -3.9 -m venv openvino_env")
os.system("openvino_env\Scripts\activate")
os.system("pip install openvino-dev")'''
from deep_translator import GoogleTranslator
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename 
from PIL import Image, ImageTk
from tkinter import scrolledtext
import PIL
import glob
from collections import namedtuple
from itertools import groupby
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
#import pytesseract
'''def backup():
    img_path = "chinese.png"
    image = Image.open(img_path)
    imgrey = image.convert('L')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(imgrey, lang='chi_sim')
    return text'''
def imageToText():
    # Directories where data will be placed
    model_folder = "model"
    data_folder = "data"
    charlist_folder = r"folder"
    
    # Precision used by model
    precision = "FP16"
    Language = namedtuple(
    typename="Language", field_names=["model_name", "charlist_name", "demo_image_name"]
    )
    chinese_files = Language(
        model_name="handwritten-simplified-chinese-recognition-0001",
        charlist_name="scut_ept.txt",
        demo_image_name="handwritten_chinese_test.jpg",
    )
    japanese_files = Language(
        model_name="handwritten-japanese-recognition-0001",
        charlist_name="japanese_charlist.txt",
        demo_image_name="handwritten_japanese_test.png",
    )
    # Select language by using either language='chinese' or language='japanese'
    language = "chinese"

    languages = {"chinese": chinese_files, "japanese": japanese_files}

    selected_language = languages.get(language)
    path_to_model_weights = Path(f'{selected_language.model_name}.bin')
    if not path_to_model_weights.is_file():
        download_command = f'omz_downloader --name {selected_language.model_name} --output_dir {model_folder} --precision {precision}'
        print(download_command)
        os.system("$download_command")
    ie = Core()
    path_to_model = path_to_model_weights.with_suffix(".xml")
    model = ie.read_model(model=path_to_model)
    # To check available device names run the line below
    # print(ie.available_devices)

    compiled_model = ie.compile_model(model=model, device_name="CPU")
    recognition_output_layer = compiled_model.output(0)
    recognition_input_layer = compiled_model.input(0)
    # Read file name of demo file based on the selected model

    file_name = selected_language.demo_image_name

    # Text detection models expects an image in grayscale format
    # IMPORTANT!!! This model allows to read only one line at time
    
    # Read image
    image = cv2.imread(filename=r"chinese.png", flags=cv2.IMREAD_GRAYSCALE)

    # Fetch shape
    image_height, image_width = image.shape

    # B,C,H,W = batch size, number of channels, height, width
    _, _, H, W = recognition_input_layer.shape

    # Calculate scale ratio between input shape height and image height to resize image
    
    scale_ratio = H / image_height

    # Resize image to expected input sizes
    resized_image = cv2.resize(
        image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA
    )
    resized_width = 0
    resized_height = 0
    # Pad image to match input size, without changing aspect ratio
    if (W - resized_image.shape[1]) > 0:
        resized_width = W - resized_image.shape[1]
        resized_image = np.pad(resized_image, ((0, 0), (int(resized_height), int(resized_width))), mode="edge")
    else:
        resized_image = cv2.resize(image, (int(image_width * (W / image_width)), int(image_height * (H / image_height))))
    # Reshape to network the input shape
    input_image = resized_image[None, None, :, :]
    plt.figure(figsize=(20, 1))
    plt.axis("off")
    plt.imshow(resized_image, cmap="gray", vmin=0, vmax=255);
    # Get dictionary to encode output, based on model documentation
    used_charlist = selected_language.charlist_name

    # With both models, there should be blank symbol added at index 0 of each charlist
    blank_char = "~"

    with open(f"scut_ept.txt", "r", encoding="utf-8") as charlist:
        letters = blank_char + "".join(line.strip() for line in charlist)
    # Run inference on the model
    predictions = compiled_model([input_image])[recognition_output_layer]
    # Remove batch dimension
    predictions = np.squeeze(predictions)

    # Run argmax to pick the symbols with the highest probability
    predictions_indexes = np.argmax(predictions, axis=1)
    # Use groupby to remove concurrent letters, as required by CTC greedy decoding
    output_text_indexes = list(groupby(predictions_indexes))

    # Remove grouper objects
    output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))

    # Remove blank symbols
    output_text_indexes = output_text_indexes[output_text_indexes != 0]

    # Assign letters to indexes from output array
    output_text = [letters[letter_index] for letter_index in output_text_indexes]
    plt.figure(figsize=(20, 1))
    plt.axis("off")
    plt.imshow(resized_image, cmap="gray", vmin=0, vmax=255)

    return "".join(output_text)
def transHelper():
    if(tImage.cget('image') == ''):
        return 'No image found.'
    else: 
        with open("chinese.txt", 'w', encoding='utf8') as f:
            f.write(imageToText())
        fileC = open("chinese.txt", encoding='utf8')
        chinese = fileC.read()
        english = GoogleTranslator(source='zh-CN', target='en').translate(chinese)
        return 'English: ' + english + '\n----------------------------\nChinese: ' + chinese
def setImage(file):
    image = Image.open(file)
    image = image.save('chinese.png')
    image1 = Image.open('chinese.png')
    ratio = 0
    if(image1.size[1] > 800):
        ratio = int(image1.size[1]/(2*(image1.size[1]/900)))
    else:
        ratio = image1.size[1]
    percent = ratio/float(image1.size[1])
    image1 = image1.resize((int(float(image1.size[0])*float(percent)),ratio))
    img = ImageTk.PhotoImage(image1)
    tImage.configure(image=img)
    tImageCaption.configure(text='Image Uploaded:')
def trans():
    tBox.delete(1.0, 'end')
    tBox.insert(tk.END, transHelper())
    if(tImage.cget('image') == ''):
        pass
    else:
        setImage('chinese.png')
    #print(english)
def upload():
    #-code goes here-
    file = askopenfilename(filetypes=[('Image Files', '*png')])
    if file is not None:
        pass
    setImage(file)
root = Tk()    
root.title("Translator Machine")
root.geometry('1000x800')
f = Frame(root)
textFrame = Frame(root)
tBox = Text(textFrame, width=50,height=25)
scrollbar = Scrollbar(textFrame)
scrollbar.pack(side=RIGHT, fill = Y)
tBox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=tBox.yview)
tImage = Label(f)
title = Label(root, text='Chinese to English Machine Learning Image Translator', font=24)
tImageCaption = Label(f, text='No Image Uploaded.')
title.pack()
tImageCaption.pack()
tImage.pack()
f.pack(pady = 5)
trBtn = Button(root, text="Translate!", command=trans, height=2, width=10)
upBtn = Button(root, text="Upload Image", command=upload, height=1, width=10)
upBtn.pack(pady = 10)
trBtn.pack(pady = 10)
tBox.pack()
textFrame.pack()
root.mainloop()
