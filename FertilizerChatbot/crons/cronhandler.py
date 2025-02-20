import logging
logger = logging.getLogger(__name__)
import os

from api.helper import train
chatbotTrain=train.ChatbotTrain()

import requests
from FertilizerChatbot.configurations.configurationhandler import CONFIG


def main(data):
    pass