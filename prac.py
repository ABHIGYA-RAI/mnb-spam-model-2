import streamlit as slt
import pickle
import string
pun = string.punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

with open('m.pkl','rb') as file:
    model = pickle.load(file)
with open('v.pkl','rb') as file:
    vec = pickle.load(file)
slt.title("Spam Detection Machine Learning Model")
user_input = slt.text_area("Enter your SMS :")
def preprocess(sms):
  sms = sms.lower()
  sms = nltk.word_tokenize(sms)
  ls = [ ]
  for i in sms:
    if i.isalnum():
      ls.append(i)
  second_ls = [ ]
  for i in ls:
    if i not in pun :
      second_ls.append(i)
  stop_words = stopwords.words('english')
  l = [ ]
  for i in second_ls:
    if i not in stop_words:
      l.append(i)
  final = [ ]
  for i in l:
    i = stemmer.stem(i)
    final.append(i)
  return " ".join(final)

if slt.button("Predict the SMS"):
    preprocessed = preprocess(user_input)
    vectorized = vec.transform([preprocessed])
    p = model.predict(vectorized)[0]
    if p == 1:
        slt.header("fake texts man..")
    elif p == 0:
        slt.header("Real !!")

