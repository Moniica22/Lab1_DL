import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error

models = ["FeedForward", "Lstm"]

def main():

    test_data = scipy.io.loadmat('Data/fake_test.mat')
    y = test_data['pred_var']

    print("\n----Experimental Results for Each Model----\n")
    for model in models:
        model_data = scipy.io.loadmat(f'{model}/predictions.mat')
        model_preds = model_data['pred_var']

        mae = mean_absolute_error(y, model_preds)
        mse = mean_squared_error(y, model_preds)
        print("-----")
        print(f"Model Name: {model}")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print("-----")



if __name__ == "__main__":
    main()