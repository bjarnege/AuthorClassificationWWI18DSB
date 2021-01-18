from flask import Flask, url_for, request, render_template, redirect, flash
import random
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
import sys
sys.path.append("../")
from RequestMapper import RequestMapper

# Initialize the Pipeline Mapper
print("Reading the models ...")
df_full_preprocessed = pd.read_pickle("../../resource/df_full_preprocessed.pkl")
pipelines_predictions = pd.read_pickle("../../resource/Pipelines/ModelPipelines.pkl")
pipelines_cluster = pd.read_pickle("../../resource/Pipelines/ClusteringPipeline.pkl")
pipelines_knn = pd.read_pickle("../../resource/Pipelines/KNearestNeighborsPipeline.pkl")

rm = RequestMapper(df_full_preprocessed, pipelines_predictions, pipelines_cluster, pipelines_knn)


# Name the clusters (index of the list equals the predicted cluster number)
text_clusters = ["Up-to-date Person", "Average Citizen", "Negation-Lover","Self-referred Author","Egocentric Person"]
numerical_clusters = ["Explanatory Author","Hobby Publisher","Daily Writer"]

# Function to create pictures out of year and age
def center_text(text, name, color=(68, 68, 68),):

    # Create the back blank background
    strip_width, strip_height = 500, 500
    img = Image.new('RGB', (strip_width, strip_height),(255, 255, 255))

    # Defintion of the font depends on the OS
    try:
        font = ImageFont.truetype("arial.ttf",150)
    except OSError:
        font = ImageFont.truetype("Arial.ttf",150)

    # Create the picture using the text printed on it
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
    draw.text(position, text, color, font=font)

    # Saving the picture to use it later
    img.save(f'./static/images/{name}.jpg')


# app = Flask(__name__) creates an instance of the Flask class called app. 
# The first argument is the name of the module or package (in this case Flask). 
# We are passing the argument __name__ to the constructor as the name of the application package. 
# It is used by Flask to find static assets, templates and so on.
app = Flask(__name__)

# As the pictures are changing after each classification, the caching of this app must be disabled
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# We set the Track Modifications to True so thatFlask-SQLAlchemy will track modifications of objects and emit signals. 
# The default is None, which enables tracking but issues a warning that it will be disabled by default in the future. 
# This requires extra memory and should be disabled if not needed.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# CREATING ROUTES
# We use Routes to associate a URL to a view function. A view function is simply a function which responds to the request. 

# This code registers the index() view function as a handler for the root URL of the application. 
# Everytime the application receives a request where the path is "/" the index() function will be invoked and the return the index.html template.
@app.route("/")
def index():
    return render_template('index.html')

# This function returns the textinput.html page which is used to input a text which should be analyzed.
@app.route("/textinput")
def textinput():
    return render_template('textinput.html')

# This function is used to pass the text input to the machine learning classification model and calculate the results if we are only using the clustering based on text features. 
@app.route("/textinput_end", methods=['GET', 'POST'])
def textinput_end():
    if request.method == 'POST':

        # Get the text input and save it globally as we will need it in another function
        global text
        text = request.form['input']

        # Get the taget variable
        global targetvariable
        targetvariable = request.form['targetvariable']

        # Continue to set the values of the other variables except the target variables
        return redirect(url_for('variableinput'))

# This function returns the variableinput.html page which is used to input the values of the additional variables.
@app.route("/variableinput")
def variableinput():

    # We are definig all possible target variables and removing the choosen one to ask for the others in the frontend.
    neededvariables = ['Age', 'Gender', 'Topic', 'Sign']
    neededvariables.remove(targetvariable)

    return render_template('variableinput.html', targetvariable=targetvariable, neededvariables=neededvariables)

# This function is used to pass the text input and the other submitted variables to the machine learning classification model and calculate the results. 
@app.route("/variableinput_end", methods=['GET', 'POST'])
def variableinput_end():
    if request.method == 'POST':

        # Define the labels as global as we need them in another function.
        global age, gender, topic, sign, clustering, knn_text
        
        # Ask for the variables which need to be presubmitted
        try:
            age = int(request.form['age'])
        except:
            age = 0

        try:
            gender = request.form['gender']
        except:
            gender = '0'

        try:
            topic = request.form['topic']
        except:
            topic = '0'

        try:
            sign = request.form['sign']
        except:
            sign = '0'
        
        # Create the stacked prediction for the target variable
        if targetvariable == "Age":
            age = int(rm.predict_weighted("age", text, sign=sign, topic=topic, gender=gender))
        elif targetvariable == "Gender":
            gender = rm.predict_weighted("gender", text, sign=sign, topic=topic, age=age)
        elif targetvariable == "Topic":
            topic = rm.predict_weighted("topic", text, sign=sign, age=age, gender=gender)
        elif targetvariable == "Sign":
            sign = rm.predict_weighted("sign", text, age=age, topic=topic, gender=gender)

        # Now we are getting results for the clustered variables
        clustering = [text_clusters[rm.transform_cluster(mode="text", text=text)], numerical_clusters[rm.transform_cluster(mode="numerical", text=text, age=age, sign=sign, gender=gender, topic=topic)]]

        # Now we are predicting a similar text via knn
        knn_text = rm.transform_knn(mode="numerical", text=text, age=age, sign=sign, gender=gender, topic=topic)

        # On the Website we use pictures for the result of age which is created now.
        center_text(str(age), 'age')
        return redirect(url_for('lastresult'))

# This function returns the calculated labels to the html templates. 
# If someone opens this site bfore inserting a text he will get autatically redirected to the textinput. 
@app.route("/lastresult")
def lastresult():
    try:
        return render_template('lastresult.html', topic=topic, age=age, clustering=clustering, gender=gender, sign=sign, text=text, knn_text=knn_text)
    except NameError:
        return render_template('textinput.html')

# This function returns the team.html page containing the information details from our team.
@app.route("/team")
def team():
    return render_template('team.html')

# Python assigns the name "__main__" to the script when the script is executed. 
# If the script is imported from another script, the script keeps it given name (e.g. app.py). 
# In our case we are executing the script. Therefore, __name__ will be equal to "__main__". 
# That means the if conditional statement is satisfied and the app.run() method will be executed.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)