import ann.custom_model_nasdaq as custom_model


def evaluateANN(hidden_layers, solution, activation_functions):
    mse = custom_model.evaluate_custom_nasdaq_model(hidden_layers, solution, activation_functions)
    return mse


def binario_a_decimal(arreglo):
    decimal = 0
    for i, bit in enumerate(reversed(arreglo)):
        decimal += bit * (2 ** i)
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
    if len(vector) != 40:
        print(vector)
        return [], []
    
    neuronasEnCapas = [ binario_a_decimal(vector[:6]),
                        binario_a_decimal(vector[6:12]),
                        binario_a_decimal(vector[12:18]) ,
                        binario_a_decimal(vector[18:24]),
                        binario_a_decimal(vector[24:30])]

    funcionesEnCapas = [determinar_activacion(vector[30:32]),
                        determinar_activacion(vector[32:34]),
                        determinar_activacion(vector[34:36]) ,
                        determinar_activacion(vector[36:38]),
                        determinar_activacion(vector[38:40])]
    
    neuronas = []
    funciones = []

    for indice, cantidad in enumerate(neuronasEnCapas):        
        if cantidad != 0:
            neuronas.append(cantidad)
            funciones.append(funcionesEnCapas[indice])

    
    return neuronas, funciones
