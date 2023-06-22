import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FUZZY:
    def __init__(self, log=False):
        self.qpa = ctrl.Antecedent(np.arange(-10, 11, 1), 'qPa')
        self.pulso = ctrl.Antecedent(np.arange(0, 201, 1), 'Pulso')
        self.resp = ctrl.Antecedent(np.arange(0, 24, 1), 'Frequência respiratória')
        self.classes = ctrl.Consequent(np.arange(1, 5, 1), 'Classe')

        self.qpa['muito baixa'] = fuzz.trimf(self.qpa.universe, [-10, -10, -5])
        self.qpa['baixa'] = fuzz.trimf(self.qpa.universe, [-10, -5, 0])
        self.qpa['média'] = fuzz.trimf(self.qpa.universe, [-5, 0, 5])
        self.qpa['alta'] = fuzz.trimf(self.qpa.universe, [0, 5, 10])
        self.qpa['muito alta'] = fuzz.trimf(self.qpa.universe, [5, 10, 10])
        
        self.pulso['muito baixa'] = fuzz.trimf(self.pulso.universe, [0, 0, 50])
        self.pulso['baixa'] = fuzz.trimf(self.pulso.universe, [0, 50, 100])
        self.pulso['média'] = fuzz.trimf(self.pulso.universe, [50, 100, 150])
        self.pulso['alta'] = fuzz.trimf(self.pulso.universe, [100, 150, 200])
        self.pulso['muito alta'] = fuzz.trimf(self.pulso.universe, [150, 200, 200])
        
        self.resp['muito baixa'] = fuzz.trimf(self.resp.universe, [0, 0, 4.4])
        self.resp['baixa'] = fuzz.trimf(self.resp.universe, [0, 4.4, 8.8])
        self.resp['média'] = fuzz.trimf(self.resp.universe, [4.4, 8.8, 13.2])
        self.resp['alta'] = fuzz.trimf(self.resp.universe, [8.8, 13.2, 17.6])
        self.resp['muito alta'] = fuzz.trimf(self.resp.universe, [13.2, 17.6, 23])
        
        self.classes['1'] = fuzz.trimf(self.classes.universe, [1, 1, 2])
        self.classes['2'] = fuzz.trimf(self.classes.universe, [1, 2, 3])
        self.classes['3'] = fuzz.trimf(self.classes.universe, [2, 3, 4])
        self.classes['4'] = fuzz.trimf(self.classes.universe, [3, 4, 4])

    def Train(self):
        rules = []

        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito baixa'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito baixa'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito baixa'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito baixa'] & self.resp['muito alta'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['baixa'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['baixa'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['baixa'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['baixa'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['média'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['média'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['média'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['média'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['média'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['alta'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['alta'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['alta'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['alta'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['alta'] & self.resp['muito alta'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito alta'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito alta'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito alta'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito alta'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito baixa'] & self.pulso['muito alta'] & self.resp['muito alta'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito baixa'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito baixa'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito baixa'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['baixa'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['baixa'] & self.resp['média'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['baixa'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['média'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['média'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['média'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['média'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['média'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['alta'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['alta'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['alta'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['alta'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['alta'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito alta'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito alta'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito alta'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito alta'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['baixa'] & self.pulso['muito alta'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito baixa'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito baixa'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito baixa'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['baixa'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['baixa'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['baixa'] & self.resp['alta'], self.classes['4']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['baixa'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['média'] & self.resp['muito baixa'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['média'] & self.resp['baixa'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['média'] & self.resp['média'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['média'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['média'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['alta'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['alta'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['alta'] & self.resp['média'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['alta'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['alta'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito alta'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito alta'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito alta'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito alta'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['média'] & self.pulso['muito alta'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito baixa'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito baixa'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito baixa'] & self.resp['muito alta'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['baixa'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['baixa'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['baixa'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['baixa'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['média'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['média'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['média'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['média'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['média'] & self.resp['muito alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['alta'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['alta'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['alta'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['alta'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['alta'] & self.resp['muito alta'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito alta'] & self.resp['muito baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito alta'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito alta'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito alta'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['alta'] & self.pulso['muito alta'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito baixa'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito baixa'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito baixa'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito baixa'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito baixa'] & self.resp['muito alta'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['baixa'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['baixa'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['baixa'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['baixa'] & self.resp['alta'], self.classes['3']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['baixa'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['média'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['média'] & self.resp['baixa'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['média'] & self.resp['média'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['média'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['média'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['alta'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['alta'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['alta'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['alta'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['alta'] & self.resp['muito alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito alta'] & self.resp['muito baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito alta'] & self.resp['baixa'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito alta'] & self.resp['média'], self.classes['1']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito alta'] & self.resp['alta'], self.classes['2']))
        rules.append(ctrl.Rule(self.qpa['muito alta'] & self.pulso['muito alta'] & self.resp['muito alta'], self.classes['1']))

        control = ctrl.ControlSystem(rules)

        self.controls = ctrl.ControlSystemSimulation(control)

    def Predict(self, X):
        y = []
        for row in X:
            self.controls.reset()
            self.controls.input['qPa'] = float(row[0])
            self.controls.input['Pulso'] = float(row[1])
            self.controls.input['Frequência respiratória'] = float(row[2])
            self.controls.compute()

            y.append(int(self.controls.output['Classe'].round()))

        return y