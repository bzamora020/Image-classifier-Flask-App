import sys
import os
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField, FileAllowed, FileRequired

# Current patch for mac. Uncomment bottom for it to work

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append('./image-classifier/omnidata/')
import classifier_single


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files/classifier'

app.config['SECRET_KEY'] = 'supersecretkey'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[FileRequired(), FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data  # First grab the file

        newname = "1" + os.path.splitext(file.filename)[1]

        filepath = (os.path.join(os.path.abspath(os.path.dirname(
            __file__)), app.config['UPLOAD_FOLDER'], secure_filename(newname)))

        file.save(filepath)  # Then save the file

        result = classifier_single.classifier(filepath)
        return render_template('response.html', form=form, result=result)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
