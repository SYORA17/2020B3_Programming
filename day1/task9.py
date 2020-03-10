import sys
import random

args = sys.argv
l = []

for word in args[1:]:
    if len(word) < 3:
        l.append(word)
    else:
        tmp = word[0] + ''.join(random.sample(word[1:-1], len(word) - 2)) + word[-1]
        l.append(tmp)

print(l)
