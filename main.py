import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

models = ["FeedForward", "Lstm", "Tcn"]

def plot_predictions(predictions, y_true, model_name):

    for i, pred in enumerate(predictions):
        plt.figure(figsize=(6.5, 3.0))
        plt.plot(pred,label = f"{model_name} Prediction", linestyle="-")
        plt.plot(y_true,label = "Ground Truth", linestyle="--")
        plt.title(f"{model_name} Recursive Prediction vs Actual Test Samples")
        plt.ylabel("Laser Measurement")
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.xlabel("Time Samples")
        plt.tight_layout()
        plt.savefig(f"prediction_plots/{model_name}/{model_name}_predictions_{i+1}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"Successfully plotted {len(predictions)} prediction{'s' if len(predictions) > 1 else ''} for model: {model_name}")
    print(f"figures saved at: prediction_plots/{model_name}")

    return

def main():

    test_data = scipy.io.loadmat('Data/Xtest.mat')
    # test_data = scipy.io.loadmat('Data/fake_test.mat')
    y = test_data['Xtest']
    os.makedirs("prediction_plots", exist_ok=True)

    print("\n----Experimental Results for Each Model----\n")
    for model in models:
        folder = f"{model}/predictions"
        prediction_files = os.listdir(folder)
        predictions = []
        os.makedirs(f"prediction_plots/{model}", exist_ok=True)

        print(f"Model Name: {model}")
        print(f"\tTotal: {len(prediction_files)} predictions to evaluate")

        
        for i, file in enumerate(prediction_files):
            model_data = scipy.io.loadmat(f'{model}/predictions/{file}')
            model_preds = model_data['pred_var']

            # Check if needed to transpose
            if model_preds.shape == (1, 200):
                model_preds = np.transpose(model_preds)
            
            predictions.append(model_preds)

            mae = mean_absolute_error(y, model_preds)
            mse = mean_squared_error(y, model_preds)
            print("-----")
            print(f"Model Name: {model} ({i+1})")
            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print("-----")

        plot_predictions(predictions=predictions, y_true=y, model_name=model)    




if __name__ == "__main__":
    main()