import json
import random

data = {}
for i in range(1, 100001):
    key = '{:06d}'.format(i)
    target = random.randint(1, 10)
    seq = [random.randint(1, 10) for _ in range(6)]
    label = target
    data[key] = {"target": target, "seq": seq, "label": label}

with open('data.json', 'w') as f:
    json.dump(data, f)