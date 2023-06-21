from cart import CART
from fuzzy import FUZZY
from nn import NN
import csv

class Testes:
    def __init__(self, arquivo_treino, arquivo_teste):
        #Le arquivo treino
        self.data_treino = []
        self.classes_treino = []
        self.gravidade_treino = []
        self.data_teste = []
        
        with open(arquivo_treino) as csvTreino:
            readerTreino = csv.reader(csvTreino, delimiter=',')

            for row in readerTreino:
                self.data_treino.append([float(row[3]),float(row[4]),float(row[5])])
                self.gravidade_treino.append(float(row[6]))
                self.classes_treino.append(row[7])
                
        with open(arquivo_teste) as csvTeste:
            readerTeste = csv.reader(csvTeste, delimiter=',')

            for row in readerTeste:
                self.data_teste.append([float(row[1]),float(row[2]),float(row[3])])
                
        self.cart = CART()
        
        self.fuzzy = FUZZY()

        self.nn = NN()
        
    def treina(self):
        self.cart.Train(self.data_treino, self.classes_treino)
        
        self.fuzzy.Train()

        self.nn.Train(self.data_treino, self.gravidade_treino)
        
    def predict_to_file(self):
        
        cart = self.cart.Predict(self.data_teste)

        with open('cart_result.txt', 'w', newline='') as cartFile:
            writer = csv.writer(cartFile, delimiter=',')
            cartList = cart.tolist()
            for i in cartList:
                writer.writerow([i])

        fuzzy = self.fuzzy.Predict(self.data_teste)

        with open('fuzzy_result.txt', 'w', newline='') as fuzzyFile:
            writer = csv.writer(fuzzyFile, delimiter=',')

            for i in fuzzy:
                writer.writerow([i])
                
        nn = self.nn.Predict(self.data_teste)

        with open('nn_result.txt', 'w', newline='') as nnFile:
            writer = csv.writer(nnFile, delimiter=',')
            nnList = nn.tolist()
            for i in nnList:
                writer.writerow([i[0]])