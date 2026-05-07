import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# I downloaded these once and added quiet=True so it doesn't spam the console every time
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    """This is the main cleaning function I wrote for the project.
    I used lemmatization instead of stemming because the task asked for it.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove everything that's not a letter
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Using NLTK's word_tokenize as required
    tokens = word_tokenize(text)
    
    # Remove common stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatization - this is what the task wanted
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)


def preprocess_data(df):
    """Just applies clean_text to the whole dataframe.
    I kept it simple because it works well.
    """
    df['cleaned_review'] = df['Review'].apply(clean_text)
    return df