from flask import Flask, url_for, request, render_template, redirect, flash
import random
from PIL import Image, ImageFont, ImageDraw


# Function to predict random values, delete when real model is implemented
def randomModel(text):
    genderClasses = ['male', 'female']
    topicClasses = ['Architecture', 'Science','Fashion','Banking','indUnk','Publishing','InvestmentBanking','Engineering','Accounting','Maritime','Religion','Museums-Libraries','RealEstate','HumanResources','Military','Chemicals','Government','Marketing','Non-Profit','Tourism','Education','LawEnforcement-Security','Automotive','Biotech','Sports-Recreation','Transportation','Advertising','Technology','Manufacturing','Consulting','Environment','Construction','Student','Agriculture','Internet','Communications-Media','Law','BusinessServices','Telecommunications','Arts']
    signClasses = ['Sagittarius','Gemini','Cancer','Pisces','Aquarius','Scorpio','Capricorn','Libra','Aries','Virgo','Taurus','Leo']
    year = random.randrange(1999, 2007)
    age = random.randrange(13, 49)
    gender = genderClasses[random.randrange(0,len(genderClasses))]
    sign = signClasses[random.randrange(0,len(signClasses))]
    topic = topicClasses[random.randrange(0,len(topicClasses))]

    return age, year, gender, sign, topic

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

# This function is used to pass the text input to the machine learning classification model and calculate the results. 
@app.route("/textinput_end", methods=['GET', 'POST'])
def textinput_end():
    if request.method == 'POST':

        # Get the text input and save it globally as we will need it in another function
        global text
        text = request.form['input']

        # Define the labels as global as we need them in another function.
        global age, year, gender, sign, topic

        # Now we are starting the text analyzation and produce a result.
        age, year, gender, sign, topic = randomModel(text)

        # On the Website we use pictures for the result of year an age which are created now.
        center_text(str(year), 'year')
        center_text(str(age), 'age')

        return redirect(url_for('lastresult'))

# This function returns the calculated labels to the html templates. 
# If someone opens this site bfore inserting a text he will get autatically redirected to the textinput. 
@app.route("/lastresult")
def lastresult():
    try:
        return render_template('lastresult.html', topic=topic, age=age, year=year, gender=gender, sign=sign, text=text)
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
    app.run(host='0.0.0.0', port=5000)