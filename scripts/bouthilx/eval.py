import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",required=True,action='append')
    parser.add_argument("--targets",required=True,action='append')
    options = parser.parse_args()

    predictions = options.predictions
    targets = [(target_path, np.load(target_path)) for target_path in options.targets]

    if len(predictions)!= len(targets):
        raise ValueError("Targets paths must match predictions paths")

    for prediction_path, (target_path, target) in zip(predictions,targets):
        y_hat= np.load(prediction_path)
        print prediction_path, target_path
        print np.mean(y_hat.argmax(1)==target)

if __name__=="__main__":
    main()
