from utils import stop_words, lemmatizer, clean_data
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model once when the server starts
model = joblib.load('classifier.pkl')

@app.route('/', methods=['POST', 'GET'])
def index():
    # Initialize the variables to avoid UnboundLocalError
    classification, color, review = None, None, None
    
    if request.method == 'POST':
        # Retrieve the review from the form and remove leading/trailing whitespace
        review = request.form.get('review', '').strip()
        
        if not review:
            classification = 'You must enter a review!'
            color = 'orange'
        else:
            try:
                # Preprocess the input text using your utility function
                clean_review = clean_data(review)
                
                # Perform prediction (expects clean_review to be a list or array)
                result = model.predict([clean_review])[0]
                
                # Logic assuming 1 is positive and 0 is negative
                if result == 1:
                    classification = f'The review "{review}" is POSITIVE!'
                    color = 'green'
                else:
                    classification = f'The review "{review}" is NEGATIVE!'
                    color = 'red'
            except Exception as e:
                # Basic error handling for prediction or cleaning issues
                classification = f"An error occurred during processing: {str(e)}"
                color = 'red'
                
    return render_template('index.html', classification=classification, color=color, review=review)

if __name__ == '__main__':
    # Run the application in debug mode for development
    app.run(debug=True)