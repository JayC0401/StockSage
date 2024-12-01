import numpy as np
from data_loader import load_data, preprocess_data, create_dataset
from model import create_model

def train_and_predict(filename):
    data = load_data(filename)
    processed_data, scaler = preprocess_data(data)
    
    time_step = 100
    X, y = create_dataset(processed_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = create_model((X.shape[1], 1))
    model.fit(X, y, epochs=100, batch_size=64, verbose=2)
    
    # Making a prediction
    test_predictions = model.predict(X[-1].reshape(1, time_step, 1))
    test_predictions = scaler.inverse_transform(test_predictions)
    return test_predictions

if __name__ == "__main__":
    result = train_and_predict('stock_data.csv')
    print("Predicted Price:", result)

