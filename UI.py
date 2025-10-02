### USER INTERFACE TO USE THE MODEL AND GET PREDICTIONS ###
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from joblib import load
def model_predict(features):
    feature_names = ["FE1", "FE2", "FE3", "FE4"]
    dataframe = pd.DataFrame(features, columns=feature_names)
    model = load('iris_model.joblib')
    my_pipeline = load('iris_pipeline.joblib')
    final_features = my_pipeline.transform(dataframe)
    prediction = model.predict(final_features)
    return prediction
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))
features = [sepal_length, sepal_width, petal_length, petal_width]
result = model_predict([features])
print("Predicted Iris Species: ", result[0])