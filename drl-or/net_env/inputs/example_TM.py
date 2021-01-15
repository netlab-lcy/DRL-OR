ret = []
for i in range(11):
    for j in range(11):
        if j == i:
            ret.append('0')
        else:
            ret.append('1')
print(' '.join(ret))
