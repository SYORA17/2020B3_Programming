import sys

args = sys.argv
n = 2

wbi_gram = []
cbi_gram = []

args = args[1:]
for i in range(len(args) - n + 1):
    wbi_gram.append(args[i:i+n])

args_c = ''.join(args)
for i in range(len(args_c) - n + 1):
    cbi_gram.append(args_c[i:i+n])


print(wbi_gram)
print(cbi_gram)
