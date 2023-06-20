import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import csv

qpa = ctrl.Antecedent(np.arange(-10, 11, 1), 'qPa')
pulso = ctrl.Antecedent(np.arange(0, 201, 1), 'Pulso')
resp = ctrl.Antecedent(np.arange(0, 23, 1), 'Frequência respiratória')
classes = ctrl.Consequent(np.arange(1, 5, 1), 'Classe')

qpa['muito baixa'] = fuzz.trimf(qpa.universe, [-10, -10, -5])
qpa['baixa'] = fuzz.trimf(qpa.universe, [-10, -5, 0])
qpa['média'] = fuzz.trimf(qpa.universe, [-5, 0, 5])
qpa['alta'] = fuzz.trimf(qpa.universe, [0, 5, 10])
qpa['muito alta'] = fuzz.trimf(qpa.universe, [5, 10, 10])

pulso['muito baixa'] = fuzz.trimf(pulso.universe, [0, 0, 50])
pulso['baixa'] = fuzz.trimf(pulso.universe, [0, 50, 100])
pulso['média'] = fuzz.trimf(pulso.universe, [50, 100, 150])
pulso['alta'] = fuzz.trimf(pulso.universe, [100, 150, 200])
pulso['muito alta'] = fuzz.trimf(pulso.universe, [150, 200, 200])

resp['muito baixa'] = fuzz.trimf(resp.universe, [0, 0, 4.4])
resp['baixa'] = fuzz.trimf(resp.universe, [0, 4.4, 8.8])
resp['média'] = fuzz.trimf(resp.universe, [4.4, 8.8, 13.2])
resp['alta'] = fuzz.trimf(resp.universe, [8.8, 13.2, 17.6])
resp['muito alta'] = fuzz.trimf(resp.universe, [13.2, 17.6, 22])

classes['1'] = fuzz.trimf(classes.universe, [1, 1, 2])
classes['2'] = fuzz.trimf(classes.universe, [1, 2, 3])
classes['3'] = fuzz.trimf(classes.universe, [2, 3, 4])
classes['4'] = fuzz.trimf(classes.universe, [3, 4, 4])

rules = []

rules.append(ctrl.Rule(qpa['baixa']&pulso['baixa']&resp['média'],classes['2']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['alta']&resp['alta'],classes['3']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['média']&resp['baixa'],classes['2']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['média']&resp['muito baixa'],classes['1']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['baixa']&resp['muito alta'],classes['2']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['muito baixa']&resp['muito alta'],classes['1']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['baixa']&resp['muito alta'],classes['2']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['muito baixa']&resp['média'],classes['1']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['muito alta']&resp['muito alta'],classes['1']))
rules.append(ctrl.Rule(qpa['média']&pulso['muito alta']&resp['alta'],classes['2']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['alta']&resp['muito alta'],classes['2']))
rules.append(ctrl.Rule(qpa['média']&pulso['baixa']&resp['média'],classes['2']))
rules.append(ctrl.Rule(qpa['média']&pulso['alta']&resp['muito alta'],classes['2']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['baixa']&resp['alta'],classes['2']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['média']&resp['alta'],classes['2']))
rules.append(ctrl.Rule(qpa['média']&pulso['muito baixa']&resp['baixa'],classes['2']))
rules.append(ctrl.Rule(qpa['média']&pulso['baixa']&resp['alta'],classes['3']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['média']&resp['alta'],classes['3']))
rules.append(ctrl.Rule(qpa['alta']&pulso['média']&resp['muito alta'],classes['3']))
rules.append(ctrl.Rule(qpa['média']&pulso['média']&resp['muito alta'],classes['3']))
rules.append(ctrl.Rule(qpa['baixa']&pulso['média']&resp['média'],classes['3']))
rules.append(ctrl.Rule(qpa['muito alta']&pulso['muito alta']&resp['baixa'],classes['1']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['muito baixa']&resp['muito baixa'],classes['1']))
rules.append(ctrl.Rule(qpa['média']&pulso['muito baixa']&resp['média'],classes['1']))
rules.append(ctrl.Rule(qpa['muito baixa']&pulso['alta']&resp['média'],classes['1']))

control = ctrl.ControlSystem(rules)

controls = ctrl.ControlSystemSimulation(control)

i = 0
certo = 0
        

with open('treino2.txt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    for row in reader:
        controls.reset()
        i += 1
        controls.input['qPa'] = float(row[3])
        controls.input['Pulso'] = float(row[4])
        controls.input['Frequência respiratória'] = float(row[5])
        controls.compute()
        
        if int(controls.output['Classe'].round()) == int(row[7]):
            certo += 1
            
        if(i == 15):
            break

print(certo/i)