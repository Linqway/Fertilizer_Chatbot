import json
import os
import html
import re

from FertilizerChatbot.configurations.configurationhandler import (
    MODEL_BASE_PATH, 
)

from . import cache
cacheEngine=cache.CacheEngine()

import logging
logger = logging.getLogger(__name__)

from . import dictionary

def de_emojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


"""
remove symbols, short forms, utls, html
Parameters:
    text: String
Returns: 
    String
"""
def sanitize_eng(text):
    VALID_TEXT_PATTERN = re.compile(r"[^?]")
    if len(re.findall(VALID_TEXT_PATTERN,text))==0:
        return None
    text = html.unescape(text) #Removing HTML charecters
    text = re.sub(r'http[s]?://\S+', "", text) #Removing links
    punctuation_dict = {
        "'s": " is",
        "n't": " not",
        "'m": " am",
        "'ll": " will",
        "'d": " would",
        "'ve": " have",
        "'re": " are",
        "&": "and",
        "₹":"rupee"
    }
    for key, value in punctuation_dict.items():
        if key in text:
            text = text.replace(key, value)

    text = de_emojify(text) #Removing emoji from text
    text = dictionary.auto_correct("eng",text)
    text = re.sub('[^A-Za-z0-9 ()\[\]\{\}]+', '', text) #Remove all special charecthers except brackets
    text = re.sub(r'^RT[\s]+', '', text)
    text = text.lower()

    text = re.sub(r'{\s+', "{", text)
    text = re.sub(r'\s+}', "}", text)

    text = re.sub(r'\(\s+', "(", text)
    text = re.sub(r'\s+\)', ")", text)

    text = re.sub(r'\[\s+', "[", text)
    text = re.sub(r'\s+\]', "]", text)

    return " ".join(text.split())
