import json
import random
import nltk

import numpy as np

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from . import dictionary
from . import preprocess


from FertilizerChatbot.configurations.configurationhandler import CONFIG

from rest_framework.exceptions import ParseError, NotFound


from . import cache
cacheEngine=cache.CacheEngine()

import logging
logger = logging.getLogger(__name__)

class ChatbotPredict:

    LANGUAGE = ""
    NOANSWER = ""
    PREDICTION_ACCURACY_THRESHOLD = ""
    IGNORE_INTENTS = []

    def load_trained_model(self,ln):

        self.LANGUAGE = ln
        self.NOANSWER = CONFIG['prediction'][ln]["noanswer"]
        self.PREDICTION_ACCURACY_THRESHOLD = CONFIG['prediction'][ln]["PREDICTION_ACCURACY_THRESHOLD"]
        self.IGNORE_INTENTS = CONFIG['prediction'][ln]["ignore_intents"]

        cachedData = cacheEngine.loadCache(ln)
        
        return (
            cachedData['model'],
            cachedData['training_dump'],
            cachedData['intents'],
            cachedData['lemmatizedWords'],
            cachedData['pattern2intents'],
        )

    def preprocess_input(self,sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words


    def bag_of_words(self,sentence,words):
        sentence_words = self.preprocess_input(sentence)
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
        return(np.array(bag))

    def clean_text(self,sentence):
        if(not sentence):
            raise ParseError()
        
        sentence = eval("preprocess.sanitize_"+self.LANGUAGE)(sentence)
        if(not sentence) : 
            return False
        sentence = dictionary.auto_correct(self.LANGUAGE,sentence)

        return sentence

    def predict_class(self,sentence,model,classes,words, data):
        bow_array = self.bag_of_words(sentence,words)
        res = model.predict(np.array([bow_array]))[0]
        results = [[i,r] for i,r in enumerate(res) if r>0.25]

        results.sort(key=lambda x: x[1], reverse=True)
        predictions = []
        for r in results:
            tag = classes[r[0]]
            probability = r[1]
            record = data['data'][tag]
            predictions.append({
                "tag": tag, 
                "probability": probability, 
                "label": record['label'], 
                "id": record['id']
            })        
        return predictions

    def direct_match(self, sentence, pattern2intents, data):
        if sentence in pattern2intents:
            direct_intent_name = pattern2intents[sentence][0]
            direct_intent = data["data"][direct_intent_name]
            accuracy = []
            accuracy.append({
                "probability": "1.0",
                "tag": direct_intent["tag"],
                "label": direct_intent["label"],
                "id": direct_intent["id"]
            })
            return accuracy
        return None


    def get_layered_match(self,sentence,model,classes,words,data,pattern2intents):
        match = "DIRECT_MATCH"
        predictions = self.direct_match(sentence, pattern2intents, data)

        if predictions is None: 
            predictions = self.predict_class(sentence,model,classes,words,data)
            match = "PREDICTION"

        return predictions,match

    def get_prediction(self,input_dict,model,classes,words,data,pattern2intents):
        sentence = self.clean_text(input_dict['sentence'])

        if(sentence) :
            predictions,prediction_type = self.get_layered_match(sentence,model,classes,words,data,pattern2intents)
            #Removing all ignore intents
            predictions = [ 
                result for result in predictions if result["tag"] not in self.IGNORE_INTENTS or
                (result["tag"] in self.IGNORE_INTENTS and float(result['probability']) == 1.0)
            ]
        else :
            predictions = []
            prediction_type = "PREDICTION"
                
        predictions_exists = True if(len(predictions)!=0) else  False

        # If prediction exists, then check if probability is below threshold
        if(predictions_exists):
            # Limiting to 3 predcitons of highest probability for suggestions
            predictions = predictions[:3] if (len(predictions) > 3) else predictions
            main_predicted_tag = predictions[0]['tag']
            main_predicted_probability = predictions[0]['probability']
            if(float(main_predicted_probability) < self.PREDICTION_ACCURACY_THRESHOLD):
                predictions_exists = False

        # if No prediction or prediction below threshold, Send noanswer tag
        if(predictions_exists == False):
            result = self.get_crt(data,self.NOANSWER)
            return {
                'type' : prediction_type,
                "probability":str(0),
                "tag": self.NOANSWER,
                "all_predictions":  predictions,
                "prediction": result['response'],
                "input_sentence": input_dict['sentence'],
                "id": result['id']
            }

        response_crt = {}
        
        response_crt = self.get_crt(data,main_predicted_tag)

        return {
            'type': prediction_type,
            "probability": predictions[0]['probability'],
            "tag": predictions[0]['tag'],
            "all_predictions" : predictions,
            'prediction':response_crt['response'],
            'id': response_crt['id'],
            "input_sentence":input_dict['sentence'],
        }

    def get_crt(self,data,tag):
        # Check if requested tag exists in database, else throw 404
        if tag in data['data'].keys():
            crt = data['data'][tag]
            return {
                "response":self.get_chatbot_response(crt),
                "id" : data['data'][tag]["id"],
                'label' : data['data'][tag]['label']
            }
        else:
            raise NotFound()


    """
    Sets chatbot Response from CRTs
    Param : Predicted CRTs, ner_entity
    returns : dictionary of responses of Request/Predicted tag
    """
    def get_chatbot_response(self,crt):

        result = {}

        if("message" in crt['responses'] 
                and len(crt['responses']['message'])!=0):
            result['message'] = random.choice(crt['responses']['message'])
        
        if("video" in crt['responses'] 
                and len(crt['responses']['video'])!=0):
            result['video'] = random.choice(crt['responses']['video'])
        
        if("form" in crt['responses'] 
                and len(crt['responses']['form']) != 0):
            result['form'] = crt['responses']['form']
        
        if("relatedFAQ" in crt['responses'] 
                and len(crt['responses']['relatedFAQ']) != 0):
            result['relatedFAQ'] = crt['responses']['relatedFAQ']

        return result


    def main(self,input_dict):
        
        if(
            input_dict['language']==None or
            (input_dict['sentence']==None and input_dict['tag'] == None)
        ):
            raise ParseError()

        #Load Model
        model,data,classes,words,pattern2intents = self.load_trained_model(input_dict['language'])
        
        #CRT Retrival by Tag
        if(input_dict['tag']!=None):
            result = self.get_crt(data,input_dict['tag'])
            response = {
                'type' : "CRT_RETRIEVAL",
                'tag': input_dict['tag'],
                'label':result['label'],
                'prediction':result['response'],
                'id': result['id'],
                "all_predictions": [{
                    "tag": input_dict['tag'],
                    "probability": str(1.0),
                    "label": result['label'],
                    "id": result['id']
                }],
            }
            return response

        else:
            return self.get_prediction(
                input_dict,
                model,
                classes,
                words,
                data,
                pattern2intents
            )