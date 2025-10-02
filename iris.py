from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import pandas as pd
from sklearn.metrics import classification_report

iris = pd.read_csv("data/iris.data")
train_set, test_set = train_test_split(iris, test_size=0.10, random_state=42, stratify=iris["LABEL"])
X_train = train_set.drop("LABEL", axis=1)
Y_train = train_set["LABEL"].copy()
X_test = test_set.drop("LABEL", axis=1)
Y_test = test_set["LABEL"].copy()
my_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy="median")),
                        ('scaler', MinMaxScaler()),
                       ])
X_train = my_pipeline.fit_transform(X_train)
X_test = my_pipeline.transform(X_test)
if __name__ == "__main__":
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)
    print(classification_report(Y_train, model.predict(X_train)))
    scores_train = cross_val_score(model, X_train, Y_train, cv=5)
    print("Training CV Accuracy: ", scores_train.mean())
    dump(model, 'iris_model.joblib')
    dump(my_pipeline, 'iris_pipeline.joblib')
    model = load('iris_model.joblib') 
    predictions = model.predict(X_test)
    print(classification_report(Y_test, predictions))
    score = cross_val_score(model, X_test, Y_test, cv=5)
    print("Test CV Accuracy: ", score.mean())
