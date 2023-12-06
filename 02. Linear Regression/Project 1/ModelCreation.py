import numpy as np
import pandas as pd
import pickle as pk
from sklearn import linear_model

def createModel(data):
    x = data['x']
    y = data['y']

    x = np.array([x]).reshape(-1,1)
    y = np.array([y]).reshape(-1,1)
    
    # train
    model = linear_model.LinearRegression()
    model.fit(x, y)

    return model

def main():

    data = pd.read_csv("example1.csv")

    model = createModel(data)

    with open('model.pkl', 'wb') as f:
        pk.dump(model, f)
        
if __name__ == '__main__':
    main()