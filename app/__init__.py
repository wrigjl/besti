import hashlib
from PIL import Image

from flask import Flask
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 5 * 1024 * 1024
app.config['SECRET_KEY'] = '0df3d46a172bcd651472f0abdf6b3c86'

from app import routes
