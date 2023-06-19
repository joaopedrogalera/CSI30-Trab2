from sklearn import tree # for decision tree models
import csv

data = []
classes = []

with open('treino2.txt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:
        data.append([row[3],row[4],row[5]])
        classes.append(row[7])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, classes)

i = 0
certo = 0

with open('treino2.txt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:
        i += 1
        r = clf.predict([[row[3],row[4],row[5]]])

        if r[0] == row[7]:
            certo += 1

print(clf.classes_)
print(clf.tree_.max_depth)
print(i)
print(certo)