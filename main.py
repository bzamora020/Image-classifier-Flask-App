import sys
import os
from flask import Flask, render_template, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired


sys.path.append(
    '/home/ersp21/Desktop/ERSP-21/codeBackup/classifier-flask-app/image-classifier/omnidata')
import classifier_single


# from image_classifier.omnidata import classifier_single

# sys.path.append(os.path.abspath("classifier-flask-app/image-classifier/omnidata")) from classifier_single import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files/classifier'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()

    if form.validate_on_submit():
        render_template('processing.html')
        flash("Image being processed\n")
        file = form.file.data  # First grab the file

        # Parsing the file name from the extension

        # filename = os.path.splitext(file.filename)[0]
        # fileExtension = os.path.splitext(file.filename)[1]

        # print("This is the filename: ", filename)
        # print("This is the file extension: ", fileExtension)

        newname = "1" + os.path.splitext(file.filename)[1]

        filepath = (os.path.join(os.path.abspath(os.path.dirname(
            __file__)), app.config['UPLOAD_FOLDER'], secure_filename(newname)))
        # print("Filepath: ", filepath, '\n')

        file.save(filepath)  # Then save the file

        result = classifier_single.classifier(filepath)
        return render_template('response.html', form=form, result=result)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
