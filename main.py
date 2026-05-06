import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import numpy as np

models = ["FeedForward", "Lstm", "Tcn"]

def main():

    # test_data = scipy.io.loadmat('Data/test_data.mat')
    test_data = scipy.io.loadmat('Data/fake_test.mat')
    y = test_data['pred_var']

    print("\n----Experimental Results for Each Model----\n")
    for model in models:
        folder = f"{model}/predictions"
        for i, file in enumerate(os.listdir(folder)):
            model_data = scipy.io.loadmat(f'{model}/predictions/{file}')
            model_preds = model_data['pred_var']

            # Check if needed to transpose
            if model_preds.shape == (1, 200):
                model_preds = np.transpose(model_preds)

            mae = mean_absolute_error(y, model_preds)
            mse = mean_squared_error(y, model_preds)
            print("-----")
            print(f"Model Name: {model} ({i+1})")
            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print("-----")



if __name__ == "__main__":
    main()