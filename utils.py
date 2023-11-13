import ann.custom_model_nasdaq as custom_model

def evaluateANN(hidden_layers, solution, activation_functions):
    mse = custom_model.evaluate_custom_nasdaq_model(hidden_layers, solution, activation_functions)
    return mse




def binario_a_decimal(arreglo):
    decimal = 0
    for i, bit in enumerate(reversed(arreglo)):
        decimal += bit * (2**i)
    return decimal


def determinar_activacion(vector):
    if vector[0] == 0 and vector[1] == 0:
        return 'sigmoid'
    elif vector[0] == 0 and vector[1] == 1:
        return 'relu'
    elif vector[0] == 1 and vector[1] == 0:
        return 'tanh'
    elif vector[0] == 1 and vector[1] == 1:
        return 'softmax'



def desnormalizar(vector):
    if len(vector) != 24:
        print(vector)
        return [],[]

    numeroEnCapa1 =  binario_a_decimal(vector[:6])    
    numeroEnCapa2 =  binario_a_decimal(vector[6:12])
    numeroEnCapa3 =  binario_a_decimal(vector[12:18])

    funcionEnCapa1 =  determinar_activacion(vector[18:20])
    funcionEnCapa2 =  determinar_activacion(vector[20:22])
    funcionEnCapa3 =  determinar_activacion(vector[22:24])

    numerosEnCapas = [numeroEnCapa1,numeroEnCapa2,numeroEnCapa3]
    funcionesEnCapas = [funcionEnCapa1,funcionEnCapa2,funcionEnCapa3]
    return numerosEnCapas, funcionesEnCapas

