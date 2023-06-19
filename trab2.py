import csv
from cart import CART

cart = CART()

data = []
classes = []
with open('testes/treino2.txt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:
        data.append([row[3],row[4],row[5]])
        classes.append(row[7])
cart.Train(data, classes)

print(cart.Test([[-8.4577,56.8384,9.2229]],['2']))