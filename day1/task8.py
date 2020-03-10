s = "Hi He Lead Because Boron Could Not Oxidize Flourine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

words = s.split(' ')
words = [word.strip('.|,') for word in words]

first_c = [1, 5, 6, 7, 8, 9, 15, 16, 19]
c = []
idx = []

for i, word in enumerate(words):
    if i+1 in first_c:
        c.append(word[0])
        idx.append(i)
    else:
        c.append(word[:2])
        idx.append(i)

d = dict(zip(c, idx))
print(d)
