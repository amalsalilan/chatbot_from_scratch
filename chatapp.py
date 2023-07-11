import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model

model = load_model('chatbot_model.h5')

intents = json.load(open('intents.json'))

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentance(sentance):
    sentence_words=nltk.word_tokenize(sentance)
    sentence_words=[lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentance,words,show_details=True):
    sentence_words=clean_up_sentance(sentance)
    bag=[0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i]=1
                if show_details:
                    print("Found in bag%s"%w)
    return(np.array(bag))


def predict_classes(sentance,model):
    p=bow(sentance,words,show_details=False)
    res=model.predict(np.array([p]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res)if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for i in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list


def getResponse(ints,intents_json):
    tag=ints[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result=random.choice(i['responses'])
            break
    return result
def chat_response(msg):
    ints=predict_classes(msg,model)
    res=getResponse(ints,intents)
    return res


import tkinter
from tkinter import *

from tkinter import Tk, Text, Scrollbar

from tkinter import Tk, Text, Scrollbar, Button


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", "end")

    if msg != "":
        ChatLog.config(state="normal")
        ChatLog.insert("end", "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chat_response(msg)
        ChatLog.insert("end", "Bot: " + res + '\n\n')

        ChatLog.config(state="disabled")
        ChatLog.yview("end")


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=False, height=False)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state="disabled")

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog["yscrollcommand"] = scrollbar.set

EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

SendButton = Button(base, text="Send", command=send)

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=265)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=276, y=401, height=90, width=100)

base.mainloop()
