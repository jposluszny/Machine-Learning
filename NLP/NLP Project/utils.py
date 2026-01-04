import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources for text processing
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_data(review):
    """
    Cleans the text by:
    1. Lowercasing and removing non-alphabetic characters
    2. Removing stopwords
    3. Reducing words to their base form (lemmatization)
    """
    
    review = re.sub('[^a-z ]', '', review.lower())
    return ' '.join([lemmatizer.lemmatize(word) for word in review.split() if word not in stop_words])

