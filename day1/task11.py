import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__))
DATA_TXT = os.path.join(ROOT_DIR, 'data.txt')

with open(DATA_TXT) as f:
    s = f.read()
    print(s)
