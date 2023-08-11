import os
import numpy
import glob
from os import listdir
from os.path import isfile, join
from flask import Flask, jsonify, render_template, request
from pathlib import Path
from PIL import Image
from Network.AI import net_obj

app = Flask(__name__, static_folder='static', )  # Для запуска  flask --app server run


@app.route('/home')
def index():
    return render_template("index.html")


def find_img_path_in_folder(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    img_files = [img for img in onlyfiles if "download-this-canvas" in img]
    img_files = [join(mypath, f) for f in img_files]
    return sorted(img_files, key=os.path.getmtime)[-1]


def guess_number(img_file):
    img = Image.open(img_file).convert("1").resize((28, 28))
    img_array = numpy.asarray(img)
    return img_array


@app.route('/guess')
def guess():
    """
    Угадывает нарисованную цифру
    """
    img_path = find_img_path_in_folder(r"C:\Users\chern\Downloads")
    predic_result = net_obj.predict_img_by_path(path=img_path)
    return render_template("guess.html", number=predic_result)
