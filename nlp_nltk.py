
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import ne_chunk
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('words')

#! 1. Load the text
text_sample = 'Nam, this is an better example setence.'

#! 2. Tokenize the text

# ? tokenize is the prcess of breakng down text into individual words or tokens.

''' if there is an issue with punkt resource,
we can try donwload it to a specific location that's easy to for nltk to find'''
nltk.download('punkt', download_dir='C:\\nltk_data')
nltk.data.path.append('C:\\nltk_data')
tokens = nltk.word_tokenize(text_sample)
print(tokens)

#! 3. Remove Stopwords

nltk.download('stopwords', download_dir='C:\\nltk_data')
nltk.data.path.append('C:\\nltk_data')

stop_words = set(stopwords.words('english'))

# 1st way to remove stopwords
filtered_tokens = []

for token in tokens:
    if token.lower() not in stop_words:
        filtered_tokens.append(token)
print(filtered_tokens)

# Second way to remove stopwords

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)

#! 4. Part of Speech tagging
# nltk.download('averaged_perceptron_tagger', download_dir='C:\\nltk_data')
# nltk.data.path.append('C:\\nltk_data')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


pos_tags = pos_tag(filtered_tokens)


#! 5. Lemmatize the Tokens

nltk.download('wordnet', download_dir='C:\\nltk_data')
nltk.data.path.append('C:\\nltk_data')

lemmatizer = WordNetLemmatizer()

filtered_tokens = [lemmatizer.lemmatize(
    token, get_wordnet_pos(pos)) for token, pos in pos_tags]


#! 6. Named Entity Recognition

'''Use the outputs from pos tags to apply entity recognition
Note that the POS tags must have the tag with it before applying the ne_chunk function'''

# nltk.download('maxent_ne_chunker')
print(ne_chunk(pos_tags))

text = 'I am working at Google'
tokens = word_tokenize(text)
pos = pos_tag(tokens)
print(ne_chunk(pos))


#! 7. Perform text analysis

''' 
Find out it quite late, turned out that with the class SentimentIntensityAnalyzer,
we do not preprocess words or sentence before computing the sentiment.
Instead, we use the raw sentence as input for the polarity_scores() to compute the sentiment
'''

# nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("What a great day! I am so happy"))
