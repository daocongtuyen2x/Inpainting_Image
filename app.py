#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import numpy as np
import base64
from werkzeug.utils import secure_filename
from models.impainting import test_biharmonic
import base64
import glob
import shutil
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    shutil.rmtree('static/uploads')
    os.makedirs('static/uploads')
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

    
 
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ', filename)
    print('link to url_for:', url_for('static', filename='uploads/' + filename))
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
@app.route('/draw/<filename>')
def draw(filename):
    return render_template('draw.html', filename = filename)

@app.route('/result/<filename>')
def inpaint(filename):
    image_path = os.path.join('static/uploads', filename)
    mask_path = os.path.join('static/uploads', 'mask.jpg')
    test_biharmonic(image_path, mask_path)
    print('done!')

    return redirect(url_for('show_result', filename=filename))

@app.route('/upload_mask',methods=["POST"])
def upload_mask():
    data = request.data
    data = data.decode("utf-8")[22:]
    base64_img_bytes = data.encode('utf-8')
    with open('static/uploads/mask.jpg', 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
    files = os.listdir('static/uploads')
    for f in files:
        if f !='mask.jpg':
            filename=f
    return redirect(url_for('inpaint', filename=filename))

@app.route('/show_result/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)


if __name__ == "__main__":
    app.run()