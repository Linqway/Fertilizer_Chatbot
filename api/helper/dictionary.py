import json
from nltk import word_tokenize
from FertilizerChatbot.configurations.configurationhandler import MODEL_BASE_PATH

import logging
logger = logging.getLogger(__name__)

from . import cache
cacheEngine=cache.CacheEngine()


"""
text auto correction using pre-defined dictionary
Parameters:
    language : String
    input_sentence : String
Returns:
    string
"""
def auto_correct(ln,input_sentence):
    if not input_sentence.strip():
        return ""

    suggestions = cacheEngine.loadCache(ln)['dictionary']
    
    tokens = word_tokenize(input_sentence)
    corrected_text = ""
    
    # preserve the token case-type
    for tkn in tokens:
        if tkn.lower() in suggestions:
            tkn = suggestions[tkn.lower()]
        corrected_text = corrected_text+tkn+" "
    return corrected_text.strip()


def save_language_dictionary(language,dict_json):
    logger.info("Saving "+language+" dictionary changes.")
    with open(MODEL_BASE_PATH+language+"/dictionary.json", 'w', encoding='utf-8') as f:
        json.dump(dict_json, f, ensure_ascii=False, indent=4)
    cacheEngine.reloadCache(language)
