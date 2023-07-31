from transformers import AutoTokenizer
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
plt.style.use('ggplot')

# read in the data
df = pd.read_csv(
    'C:\\Users\\brian\\My Drive\\WF Summer\\Analytics Course\\Code Repo\\NLP\\NLP\\rv_amz.csv')

# for the sake of efficiency, we reduce the data to just 10,000 rows

df = df.head(10000)

# recheck the shape

df['Text'][0]

# EDA of Review Score


df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars')\
    .set_xlabel('Review Star')
plt.show()

# Basic NLTK

example = df['Text'][50]

print(example)

# tokenize
tokens = nltk.word_tokenize(example)

# part of speech

tagged = nltk.pos_tag(tokens)

# ne_chunk
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

#! VADER sentiment scoring


sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am so happy')

sia.polarity_scores('This is the worst thing ever')

# Create a new column in the data to assess the sentiment of review using VADER
sentiment_type = [('vader_neg', 'neg'),
                  ('vader_neu', 'neu'), ('vader_pos', 'pos')]

for i, j in sentiment_type:
    df[i] = [sia.polarity_scores(review)[j] for review in df['Text']]


# Plot VADERS result

sns.barplot(data=df, x='Score', y='Sentiment')
plt.title('Compound Score by Amazon Star Review')
plt.show()

#! RoBERTa pretrained model


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#! Compare performance of VADER and RoBERTa model

# 1. VADER result
print(example)

sia.polarity_scores(example)

# 2. RoBERTa result

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

''' 
Create a function to convert output of RoBERTa model to directly compare with VADER result
'''


def polarity_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


polarity_roberta(example)


roberta_sentiment_type = ['roberta_neg', 'roberta_neu', 'roberta_pos']

a = df.head(10)

for i in roberta_sentiment_type:
    try:
        a[i] = [polarity_roberta(review)[i] for review in a['Text']]
    except RuntimeError:
        print('Broke')

a
