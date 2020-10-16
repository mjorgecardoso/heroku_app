from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

import numpy as np  
from tensorflow.keras.models import load_model
import joblib



def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    p1 = sample_json['param1']
    p2 = sample_json['param2']
    p3 = sample_json['param3']
    p4 = sample_json['param4']
    
    pjoint = [[p1,p2,p3,p4]]
    
    pjoint = scaler.transform(pjoint)
    
    classes = np.array(['Healthy', 'Non-Covid ILI', 'COVID'])
    
    class_ind = model.predict_classes(pjoint)
    
    return classes[class_ind][0]



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
my_model = load_model("final_iris_model.h5")
my_scaler = joblib.load("iris_scaler.pkl")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class FlowerForm(FlaskForm):
    sep_len = TextField('Parameter 1')
    sep_wid = TextField('Parameter 2')
    pet_len = TextField('Parameter 3')
    pet_wid = TextField('Parameter 4')

    submit = SubmitField('Predict')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['param1'] = form.param1.data
        session['param2'] = form.param2.data
        session['param3'] = form.param3.data
        session['param4'] = form.param4.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():

    content = {}

    content['param1'] = float(session['param1'])
    content['param2'] = float(session['param2'])
    content['param3'] = float(session['param3'])
    content['param4'] = float(session['param4'])

    results = return_prediction(model=my_model,scaler=my_scaler,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)
