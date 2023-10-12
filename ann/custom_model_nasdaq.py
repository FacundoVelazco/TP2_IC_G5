import ann.utils as utils
from sklearn.metrics import mean_squared_error
from keras import layers, models, activations


def evaluate_custom_nasdaq_model(hidden_layers, solution, activation_functions):
    path = 'ann/nasdaq-index-365.json'

    raw_data = utils.getData(path)

    max_open = max(item["open"] for item in raw_data)
    min_open = min(item["open"] for item in raw_data)

    normalized_data = utils.normalize(raw_data, max_open, min_open)

    ((train_data_4days, train_labels_4days), (validation_data_4days, validation_labels_4days),
     (test_data_4days, test_labels_4days)) = utils.genTrainData4DaysBf(normalized_data)

    def build_model_regression(input_data_shape, hidden_layers, neurons_number, activation_functions):
        model = models.Sequential()
        model.add(layers.Dense(20, activation=activations.tanh, input_shape=[input_data_shape, ]))
        for i in range(0, hidden_layers):
            model.add(layers.Dense(neurons_number[i], activation=activation_functions[i]))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        return model

    model = build_model_regression(len(train_data_4days[0]), hidden_layers, solution, activation_functions)

    history = model.fit(train_data_4days, train_labels_4days, epochs=200,
                        validation_data=(validation_data_4days, validation_labels_4days),
                        verbose=False)

    predicted_values = model.predict(test_data_4days)
    predicted_values = utils.desnormalizeList(predicted_values, max_open, min_open)
    real_values = utils.desnormalize(test_labels_4days, max_open, min_open)

    mse = mean_squared_error(real_values, predicted_values)

    return mse
