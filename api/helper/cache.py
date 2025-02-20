import json
import os

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import pickle

# Loading Trained model
from keras.models import load_model

from FertilizerChatbot.configurations.configurationhandler import MODEL_BASE_PATH, CONFIG

import logging
logger = logging.getLogger(__name__)
from django.core.cache import cache # This is the memcache cache.

class CacheEngine:

    CACHE = {}

    def checkIfFileExists(self,file):
        if (
            os.path.isfile(file) and 
            os.access(file, os.R_OK) and 
            os.stat(file).st_size != 0
        ):
            return True
        else :
            return False
    
    def swapKeyValue(self,data):
        return dict([(value, key) for key, value in data.items()])
    
    def reloadCache(self,ln):
        logger.info("Reloading Cache in Memory")
        self.CACHE[ln] = {}
        file_location = MODEL_BASE_PATH + ln +'/models/chatbot/'

        # First load dictionary since, It is required for preprocessing and is not
        # dependent on prediction cache
        if(self.checkIfFileExists(MODEL_BASE_PATH+ln+"/dictionary.json")):
            dictionary_cache = self.loadDictionary(ln)
        else:
            self.CACHE.pop(ln)
            raise PreconditionFailed()
        
        if(self.checkIfFileExists(file_location+'chatbot_model.hdf5')):
            self.CACHE[ln] = {
                "model" : load_model(file_location+'chatbot_model.hdf5'),
                "training_dump" : json.loads(open(file_location+"basicQA.json" ,encoding="utf-8").read()),
                "intents" : json.loads(open(file_location+'classes.json' ,encoding="utf-8").read()),
                "lemmatizedWords" : json.loads(open(file_location+'lemmatizedWords.json' ,encoding="utf-8").read()),
                "pattern2intents" : json.loads(open(file_location+'pattern2intents.json' ,encoding="utf-8").read()),
                "dictionary" : dictionary_cache,
                # "cropType" : self.swapKeyValue(json.loads(open(file_location+'cropType.json' ,encoding="utf-8").read())),
                # "soilType" : self.swapKeyValue(json.loads(open(file_location+'soilType.json' ,encoding="utf-8").read())),
                # "fertilizers" : self.swapKeyValue(json.loads(open(file_location+'fertilizers.json' ,encoding="utf-8").read())),
            }
        else:
            self.CACHE[ln]["dictionary"] = dictionary_cache

        cache.clear()
        return True

    def loadDictionary(self,ln):
        language_dictionary = json.loads(open(MODEL_BASE_PATH+ln+"/dictionary.json" ,encoding="utf-8").read())
        suggestions = {}
        for actual, mistyped in language_dictionary.items():
            if type(mistyped) is not bool:

                # handle comma separated string or numbers
                if type(mistyped) is str or type(mistyped) is int:
                    mistyped = str(mistyped).split(",")

                for mis in mistyped:
                    mis=str(mis).lower() # string key in case of other types
                    suggestions[str(mis).lower()] = actual

        return suggestions

    
    def loadCache(self,ln):
        if(CONFIG['training']['isCachingEnabled'] == False):
            self.reloadCache(ln)
        else:
            if (
                ln not in self.CACHE.keys() or
                not self.CACHE
            ):
                logger.info("Cache is Empty.Builing Cache in Memory")
                self.reloadCache(ln)
        
        return self.CACHE[ln]