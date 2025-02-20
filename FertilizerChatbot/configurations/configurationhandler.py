from django.conf import settings
import os
from pathlib import Path
import json

BASE_APP_PATH = str(Path(__file__).resolve().parent.parent.parent)

# Path to resources directory
MODEL_BASE_PATH = getattr(settings, 'MODEL_BASE_PATH', BASE_APP_PATH + '/resources/')

# Path to Configuration
CONFIG_FILE_PATH = BASE_APP_PATH + '/config.json'
CONFIG = json.loads(open(CONFIG_FILE_PATH ,encoding="utf-8").read())
CONFIG = getattr(settings, 'CONFIG', CONFIG)