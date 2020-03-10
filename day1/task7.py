s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

l = []
prev = True
cnt = 0
for c in s:
    if c == ' ' or c == ',' or c == '.':
        if prev:
            l.append(cnt)
        prev = False
        cnt = 0
    else:
        prev = True
        cnt += 1

print(l)
