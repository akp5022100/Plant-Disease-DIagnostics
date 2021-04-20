from flask import Flask,render_template,url_for,flash,send_from_directory,request,redirect
from werkzeug.utils import secure_filename
from ml import GradCam
import os
import PIL
from PIL import ImageTk
import os
from pathlib import Path
import fastai
from fastai import *
from fastai.vision import *
import numpy as np
path = Path('PlantVillage') 
np.random.seed(42)
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from fastai.tabular import *

UPLOAD_FOLDER = 'C:\\Users\\Abhishek\\Desktop\\MTP2\\uploads'
MODEL_FOLDER = 'C:\\Users\\Abhishek\\Desktop\\MTP2\\models'

path=Path('C:\\Users\\Abhishek\\PlantVillage')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

data=fastai.vision.data.ImageDataBunch.from_folder(path, 
                                  train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=fastai.vision.transform.get_transforms(), 
                                  size=450, 
                                  num_workers=4, 
                                  bs = 16) \
        .normalize(imagenet_stats)


alexNet=load_learner(MODEL_FOLDER, 'densenet_121.pkl')
vggNet=load_learner(MODEL_FOLDER, 'densenet_121.pkl')
resNet=load_learner(MODEL_FOLDER, 'resnet.pkl')
denseNet=load_learner(MODEL_FOLDER, 'densenet_121.pkl')
#learn.load('stage1-300-8epochs')




app=Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['RESULTS_FOLDER'] = RESULT_FOLDER
app.config["CACHE_TYPE"] = "null"

def get_analysis(image_path,architecture):
    img = open_image(image_path)
    label=None
    prob=0
    filename=image_path.split('\\')[-1]
    if architecture=='AlexNet':
        gcam = GradCam.from_one_img(alexNet,img)
        label,prob=gcam.plot(filename)
    elif architecture=='VGGNet':
        gcam = GradCam.from_one_img(vggNet,img)
        label,prob=gcam.plot(filename)
    elif architecture=='ResNet':
        gcam = GradCam.from_one_img(resNet,img)
        label,prob=gcam.plot(filename)
    else:
        gcam = GradCam.from_one_img(denseNet,img)
        label,prob=gcam.plot(filename)
    return label,prob


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('home.html')

@app.route('/show/<filename>',methods=['GET', 'POST'])
def uploaded_file(filename):
    if request.method=='POST':
        architecture = request.form['Architectures']

        return redirect(url_for('analyze_file',filename=filename,architecture=architecture))

    filename = 'http://127.0.0.1:5000/uploads/' + filename
    return render_template('display.html', filename=filename)



@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)




@app.route('/analyze/<filename>/<architecture>')
def analyze_file(filename,architecture):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    label,prob=get_analysis(path,architecture)
    f='http://127.0.0.1:5000/uploads/gradcam_'+filename 
    return render_template('analyze.html',f=f,architecture=architecture,label=label,prob=prob)


@app.route('/about')
def about():
    return render_template('about.html',title='About')


@app.route('/register')
def register():
	form=RegistrationForm()
	return render_template('register.html',title='Register',form=form)

@app.route('/login')
def login():
	form=LoginForm()
	return render_template('login.html',title='Login',form=form)

@app.route('/alexnet')
def alexnet():
    return render_template('alexnet.html')

@app.route('/resnet')
def resnet():
    return render_template('resnet.html')

@app.route('/vggnet')
def vggnet():
    return render_template('vggnet.html')

@app.route('/densenet')
def densenet():
    return render_template('densenet.html')

@app.route('/compareArchitecture')
def compareArchitecture():
    return render_template('compareArchitecture.html')


if __name__=='__main__':
	app.run(debug=True)