from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn import tree # for decision tree models

class CART:
    def Train(self, X, y):

        # Fit the model
        self.model = tree.DecisionTreeClassifier(criterion='entropy', 
                                            max_depth=5,
                                            min_samples_leaf=5
                                          )
        self.clf = self.model.fit(X, y)

    def Predict(self, X):
        return self.model.predict(X)

    def Test(self, X, y):
        pred = self.model.predict(X)

        print('*************** Evaluation on Test Data ***************')
        score_te = self.model.score(X, y)
        print('Accuracy Score: ', score_te)
        # Look at classification report to evaluate the model
        print(classification_report(y, pred))
        print('--------------------------------------------------------')
        print("")