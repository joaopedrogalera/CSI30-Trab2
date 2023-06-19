from testes import Testes
import sys
import os

def main(arquivo_teste):
    if not os.path.isfile(arquivo_teste):
        raise Exception('arquivo ' + arquivo_teste + ' não existe')

    teste = Testes('treino_sinais_vitais_com_label.txt', arquivo_teste)
    
    teste.treina()
    teste.predict_to_file()
    
    
main(sys.argv[1] if len(sys.argv) > 1 else 'teste1.txt')