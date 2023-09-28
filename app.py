from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
with open('Spam_Email_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
def preprocess(text):
    ps = PorterStemmer()
    ### only considering the alphabet
    text = re.sub('[^a-zA-Z]', ' ',text)
    
    #### lower case
    text = text.lower()
    text = text.split()

    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    
    return text

@app.route('/', methods=['GET', 'POST'])
def classify_email():
    if request.method == 'POST':
        email_text = request.form['email_text']
        
        # Perform any necessary data preprocessing here
        email_text = preprocess(email_text)
        
        # Make a prediction using the model
        prediction = model.predict([email_text])
        
        # Interpret the prediction
        if prediction == 0:
            result = 'Not Spam'
        else:
            result = 'Spam'
        
        return render_template('index.html', result=result, email_text=email_text)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
