import pandas as pd

from FertilizerChatbot.configurations.configurationhandler import MODEL_BASE_PATH
import json

from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize,LabelEncoder,label_binarize,RobustScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

def swapKeyValue(data):
    return dict([(value, key) for key, value in data.items()])

def buildInput(data,file_location):
    cropType = swapKeyValue(json.loads(open(file_location+'cropType.json' ,encoding="utf-8").read()))
    soilType = swapKeyValue(json.loads(open(file_location+'soilType.json' ,encoding="utf-8").read()))

    return [
        data['temparature'],data['humidity'],data['moisture'],
        int(soilType[data['soilType']]),int(cropType[data['cropType']]),
        data['nitrogen'],data['potassium'],data['phosphorous'],
    ]


def load_pickle(ln,name):
    tokenizer_f = open(MODEL_BASE_PATH + ln +'/models/chatbot/'+name, "rb")
    tokenizer = pickle.load(tokenizer_f)
    tokenizer_f.close()
    return tokenizer

def getFertilizerName(data,ln):

    file_location = MODEL_BASE_PATH + ln +'/models/chatbot/'

    data = buildInput(data,file_location)

    LR = LogisticRegression()
    Fertilizer_Name = json.loads(open(file_location+'fertilizers.json' ,encoding="utf-8").read())
    LR = load_pickle(ln,'LogisticRegression.pkl')
    data = np.array([data])
    prediction = LR.predict(data)
    return Fertilizer_Name[str(prediction[0])]

def getFertilizerType(data,ln):
    fp = pd.read_csv(MODEL_BASE_PATH + ln +"/models/chatbot/FertilizerPrediction.csv")

    sc = fp.query('SoilType == "'+data['soilType']+'" and CropType=="'+data['cropType']+'"')
    if sc.empty:
        return { "message" : "Crop Not Suitable for this Soil" }
    
    name = getFertilizerName(data,ln)
    return {'message' : name+" is Suitable for your Crop" }