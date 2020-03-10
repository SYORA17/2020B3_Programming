import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__))
DATA_TXT = os.path.join(ROOT_DIR, 'data.txt')

docs = []
terms = set()
with open(DATA_TXT) as f:
    s = [s.strip() for s in f.readlines()]
    for l in s:
        l = l.split('と')
        docs.append(l)
        for fruit in l:
            terms.add(fruit)

print("docs : ", docs)
print("terms: ", list(terms))
