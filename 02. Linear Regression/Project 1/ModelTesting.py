import pandas as pd
import pickle as pk
import numpy as np

def Predictions(inputData):
    model = pk.load(open("model.pkl", "rb"))

    prediction = model.predict(inputData)

    return prediction

prediction = Predictions([[6]])

print(prediction)