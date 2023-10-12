import ann.custom_model_nasdaq as custom_model


def evaluateANN(hidden_layers, solution, activation_functions):
    mse = custom_model.evaluate_custom_nasdaq_model(hidden_layers, solution, activation_functions)
    return mse
