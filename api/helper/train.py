import json
import re

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords 
STOPWORDS = set(stopwords.words('english'))

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import pathlib

from FertilizerChatbot.configurations.configurationhandler import (
    MODEL_BASE_PATH,
    CONFIG
)

from rest_framework.exceptions import ParseError

from pytz import timezone 
from datetime import datetime


import logging
logger = logging.getLogger(__name__)

from . import cache
cacheEngine=cache.CacheEngine()


from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
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

class ChatbotTrain:

    LANGUAGE = ""
    CURRENT_TIME = ""
    DEV_DIRECTORY = ""
    
    def setupTraningResources(self,ln):
        self.LANGUAGE = ln
        self.CURRENT_TIME = datetime.now(timezone("Asia/Kolkata")).strftime(
            CONFIG['training']['datetime_format']
        )
        self.DEV_DIRECTORY = MODEL_BASE_PATH+ln+"/models/chatbot/dev/"+self.CURRENT_TIME+"/"
        pathlib.Path(self.DEV_DIRECTORY).mkdir(parents=True, exist_ok=True) 
    
    
    def load_data(self):
        data_file = MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/basicQA.json'
        data = json.loads(open(data_file ,encoding="utf-8").read())
        return data
    

    """
    Removes duplicate patterns case insensitive
    Parameters:
        data    : Dict dataset object      
        language: String
    Returns:
        Dict
    """
    def data_deduplication(self,data):
        duplicates = {}
        for key in data['data']:
            values = data['data'][key]
            unq_patterns = []
            for pattern in values['patterns']:
                noramlized_pattern = re.sub(r"[\!\?,]", "",pattern.lower())
                if noramlized_pattern not in duplicates:
                    unq_patterns.append(noramlized_pattern)
                    duplicates[noramlized_pattern] = True
            data['data'][key]['patterns'] = unq_patterns

        return data    

    def preprocess_input(self,data): 
        words=[]
        classes = []
        documents = []
        for row in data['data'].values():
            if row["tag"].startswith("@"):
                logger.info("Excluding training for intent " + row["tag"])
                continue

            for pattern in row['patterns']:
                pattern =  " ".join([word for word in pattern.split() if word not in STOPWORDS])
                #tokenizing each word
                tokenized_words = nltk.word_tokenize(pattern)
                words.extend(tokenized_words)

                #adding documents in the corpus
                documents.append((tokenized_words, row['tag']))

                # adding to classes list
                if row['tag'] not in classes:
                    classes.append(row['tag'])
        # lemmaztizing and lower each word and remove duplicates
        words = [lemmatizer.lemmatize(w.lower()) for w in words ]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        # Adding word and tags list to json file for prediction
        with open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/lemmatizedWords.json', 'w', encoding='utf-8') as f:
            json.dump(words, f, ensure_ascii=False, indent=4)

        with open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/classes.json', 'w', encoding='utf-8') as f:
            json.dump(classes, f, ensure_ascii=False, indent=4)
        
        logger.info("Training data preprocessed")
        return words,classes,documents


    def create_training_data(self,words,classes,documents):
        # creating training data
        training = []
        # creating an empty list for output and filling it with zero
        output_empty = [0] * len(classes)
        # training set for bag of words for each sentence
        for doc in documents:
            # initializing bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemmatizing each word - creating base word, in attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1, if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        training = np.array(training,dtype=object)

        x = list(training[:,0])
        y = list(training[:,1])
        x = np.array(x)
        y = np.array(y)

        train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.05, random_state=0) #test_size=0.2 misses more training pattern

        logger.info("Training data configured for training")
        return train_x, test_x, train_y, test_y

    def preparePrerequisites(self,data):
        data = self.data_deduplication(data)
        words,classes,documents = self.preprocess_input(data)
        train_x, test_x, train_y, test_y = self.create_training_data(words,classes,documents)
        return train_x, test_x, train_y, test_y

    
    def plot_evaluation_graph(self,history):
        for attribute,details in CONFIG['training']['modelEvaluationGraph'].items():
            plt.plot(history.history[details['attributes'][0]['attribute']])
            plt.plot(history.history[details['attributes'][1]['attribute']])
            plt.title(details['title'])
            plt.ylabel(details['y_axis_label'])
            plt.xlabel(details['x_axis_label'])
            plt.legend([details['attributes'][0]['label'],details['attributes'][1]['label']], loc='upper left')
            plt.savefig(self.DEV_DIRECTORY+details['filename'])
            plt.clf()


    def evaluate_accuracy(self,test_x,test_y):
        best_model = load_model(
            MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/chatbot_model.hdf5'
        )
        test_loss, test_acc = best_model.evaluate(test_x, test_y, verbose=2)
        file_path = self.DEV_DIRECTORY + "accuracy.json"
        with open(file_path,'w',encoding='utf-8') as f:
            json.dump(
                {"accuracy" : test_acc, "loss" : test_loss}, 
                f, ensure_ascii=False, indent=4
            )
    
    # Creating model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of data to predict output tag with softmax
    def train_model(self,train_x, test_x, train_y, test_y):
        logger.info("Started Training "+self.LANGUAGE+" Model")

        # Define model and layers
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))
        
        # Compiling model. 
        # Stochastic gradient descent with Nesterov accelerated gradient gives 
        # good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        # setup logger to log training data
        csv_logger = CSVLogger(self.DEV_DIRECTORY+"trainingLog.csv", append=True)

        logger.info("Fitting and saving the model")
        
        # setup Checkpoints
        checkpoint = ModelCheckpoint(
            MODEL_BASE_PATH+self.LANGUAGE+"/models/chatbot/chatbot_model.hdf5", 
            monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='auto', 
            period=1, save_weights_only=False
        )

        #fitting and saving the model 
        hist = model.fit(
            train_x, train_y,
            validation_data=(test_x, test_y), epochs=200    , 
            batch_size=10, verbose=1, callbacks=[csv_logger,checkpoint]
        )

        # Evaluate Accuary and plot graph
        self.evaluate_accuracy(test_x,test_y)
        self.plot_evaluation_graph(hist)

        logger.info("Finished Training "+self.LANGUAGE+" Model")
        return True
    
    def pattern2intents(self,data):
        pattern2intents = {}
        for intent_name in data["data"]:
            intent = data["data"][intent_name]
            if "patterns" in intent:
                for pattern in intent["patterns"]:
                    if pattern not in pattern2intents:
                        pattern2intents[pattern] = [intent_name]
                    else:
                        pattern2intents[pattern].append(intent_name)
                        logger.info ("[Warning] Multiple intent matches pattern: "+ str(pattern) + " - "+ str(pattern2intents[pattern]))
        
        with open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/pattern2intents.json', 'w', encoding='utf-8') as f:
            json.dump(pattern2intents, f, ensure_ascii=False, indent=4)



    def buildFertilizerTrainingModel(self):
        fert=pd.read_csv(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/FertilizerPrediction.csv')
        fert.isnull().sum()
        fert.nunique()
        fert.dtypes[fert.dtypes =='int64']
        fert.dtypes[fert.dtypes =='object']

        fert['Fertilizer Name'].unique()
        fert['Fertilizer Name'].value_counts()
        fert['CropType'].value_counts()
        fert['SoilType'].value_counts()

        from sklearn.preprocessing import LabelEncoder
        
        soil_type_label_encoder = LabelEncoder()
        fert["SoilType"] = soil_type_label_encoder.fit_transform(fert["SoilType"])
        crop_type_label_encoder = LabelEncoder()
        fert["CropType"] = crop_type_label_encoder.fit_transform(fert["CropType"])
        croptype_dict = {}
        for i in range(len(fert["CropType"].unique())):
            croptype_dict[i] = crop_type_label_encoder.inverse_transform([i])[0]

        with open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/cropType.json', 'w', encoding='utf-8') as f:
            json.dump(croptype_dict, f, ensure_ascii=False, indent=4)

        soiltype_dict = {}
        for i in range(len(fert["SoilType"].unique())):
            soiltype_dict[i] = soil_type_label_encoder.inverse_transform([i])[0]

        with open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/soilType.json', 'w', encoding='utf-8') as f:
            json.dump(soiltype_dict, f, ensure_ascii=False, indent=4)

        fert_label= pd.DataFrame(fert['Fertilizer Name'])

        le2 = preprocessing.LabelEncoder()
        enc_label = fert_label.apply(le2.fit_transform)
        fert['Fertilizer'] = enc_label

        Fertilizer_Name = {}
        for i in range(len(fert["Fertilizer"].unique())):
            Fertilizer_Name[i] = le2.inverse_transform([i])[0]
        print(Fertilizer_Name)

        with open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/fertilizers.json', 'w', encoding='utf-8') as f:
            json.dump(Fertilizer_Name, f, ensure_ascii=False, indent=4)

        fert_multi= fert.drop(['Fertilizer Name'],axis=1)


        y_train_multi= fert_multi[['Fertilizer']]
        X_train_multi= fert_multi.drop(labels=['Fertilizer'], axis=1)

        smote = SMOTE()
        X_train_multi,y_train_multi = smote.fit_resample(X_train_multi,y_train_multi)
        X_train, X_test, y_train, y_test = train_test_split(X_train_multi, y_train_multi, test_size=0.2, random_state=42, shuffle=True)


        rs = RobustScaler()
        rs.fit(X_train)
        X_train_scaled =rs.transform(X_train)
        X_test_scaled = rs.transform(X_test)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit(y_train)
        y_train_encoded = le.transform(y_train)
        y_test_encoded = le.transform(y_test)

        from sklearn.linear_model import LogisticRegression
        LR = LogisticRegression()
        LR.fit(X_train_scaled,y_train_encoded)
        print('Training accuracy: ', LR.score(X_train_scaled,y_train_encoded))  

        LR_Model_pkl = open(MODEL_BASE_PATH+self.LANGUAGE+'/models/chatbot/LogisticRegression.pkl', 'wb')
        pickle.dump(LR, LR_Model_pkl)
        LR_Model_pkl.close()


    def main(self,ln):
        if(ln==None):
            raise ParseError()
        
        # Setup resources directory
        self.setupTraningResources(ln)

        data = self.load_data()

        # Prepare data
        train_x, test_x, train_y, test_y = self.preparePrerequisites(data)
        
        # Train Chatbot Model
        isTrained = self.train_model(train_x, test_x, train_y, test_y)
        
        # Relod Cache after models are trained.
        if isTrained :
            self.pattern2intents(data)
            self.buildFertilizerTrainingModel()
            return cacheEngine.reloadCache(ln)