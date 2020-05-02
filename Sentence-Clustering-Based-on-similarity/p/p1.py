
data = list()
for line in open('DataP_8.txt','r'):
    data.append(line)



from difflib import SequenceMatcher

def getRatio(a, b):
    return SequenceMatcher(None, a, b).ratio()

treshold     = 0.95
minGroupSize = 1

from itertools import combinations

paired = { c:{c} for c in data }
for a,b in combinations(data,2):
    if getRatio(a[18:38],b[18:38]) < treshold: continue
    paired[a].add(b)
    paired[b].add(a)

groups    = list()
ungrouped = set(data)
while ungrouped:
    bestGroup = {}
    for city in ungrouped:
        g = paired[city] & ungrouped
        for c in g.copy():
            g &= paired[c] 
        if len(g) > len(bestGroup):
            bestGroup = g
    if len(bestGroup) < minGroupSize : break  # to terminate grouping early change minGroupSize to 3
    ungrouped -= bestGroup
    groups.append(bestGroup)
print('Total number of families formed from list :',len(groups))
print('\n')
print('Sample families are \n',groups[0])
'''print(groups[1])
print(groups[2])'''
print(groups)
