from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

class CART:
    def __init__(self, log=False):
        self.log = log

    def Train(self, X, y):

        # Fit the model
        self.model = tree.DecisionTreeClassifier(criterion='entropy', 
                                            max_depth=7,
                                            min_samples_leaf=5
                                          )
        self.clf = self.model.fit(X, y)

    def Predict(self, X):
        return self.model.predict(X)

    def Test(self, X, y):
        pred = self.model.predict(X)

        if self.log:
            print('*************** Evaluation on Test Data ***************')
            score = self.model.score(X, y)
            print('Accuracy Score: ', score)
            # Look at classification report to evaluate the model
            print(classification_report(y, pred))
            print('--------------------------------------------------------')
            print("")
        return score