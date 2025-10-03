[](https://bouqs.com/blog/wp-content/uploads/2021/11/iris-flower-meaning-and-symbolism.jpg)
# Iris Flower Classification
The **iris flower classification** is popularly known as the **Hello World!** of Machine Learning.. I used the **Iris dataset** provided by **UCI Machine Learning Repository** 
[Download Iris dataset](https://archive.ics.uci.edu/static/public/53/iris.zip)
The dataset consists of 4 features : 
- sepal length
- sepal width
- petal length
- petal width
and 3 labels : 
- Iris-Sentosa
- Iris-Versicolor
- Iris-Virginica
<br/>For my convinience, i mentioned the features column as "FE1, FE2, FE3, FE4" and labels' column as "LABEL" in my original data [iris.data](https://archive.ics.uci.edu/static/public/53/iris.zip).

![Dataset](https://github.com/debayanroy676/iris_machine_learning/blob/master/graphs/Featurewise%20histogram.jpg?raw=true)

The above histogram represents the total number of flowers with feature sepal-length, sepal-width, petal-length and petal-with.

# Key Points to consider
I have used **sci-kit learn** and **pandas** module of python programming language and elaborated how i implemented this project below :
- The source code of creation/training/testing of the model is available at [iris.py](https://github.com/debayanroy676/iris_machine_learning/blob/master/iris.py).
- The second file [UI.py](https://github.com/debayanroy676/iris_machine_learning/blob/master/UI.py) serves as the CLI User-Interface.
- I converted the csv data to a pandas dataframe
- Then I carried out train-test split, to get separate training and testing data
- The training/testing datas are put into a pipeline scaled using MinMaxScaler and any discrepency in the given data is handled by SimpleImputer that assigns "median value" of a particular feature to handle any missing data.
- I selected KNeighborsClassifier as our model and trained it with my training set and later tested it. 
- I used 5 fold cross-validation-score to ensure best accuracy and also to ensure my model doesn't overfits the data.
---
# Usage
```bash
git clone https://github.com/debayanroy676/iris_machine_learning.git
cd iris_machine_learning
#if sklearn and pandas are not installed
pip install scikit-learn
pip install pandas
#else directly run
python iris.py
python UI.py
```
# Why KNeighborsClassifier

I used KNeighborsClassifier with n_neighbors=3, which finds out 3 nearest labels with respect to our input data and gives prediction based on the majority label encountered,
This is illustrated with a very simple diagram below :
![KNN](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/ef/3a/KNN.png)

Another reason for this approach is the tidyness of the iris-dataset that is illustrated with the help of matplotlib.seaborn scatter matrix
![Scatter Matrix](https://github.com/debayanroy676/iris_machine_learning/blob/master/graphs/Scatter%20Plot.jpg?raw=true)
Since we see the data is not much of a mess, I used KNeighborsClassifier. I thought of using RandomForestClassifier, but it is not ideal for a small dataset like the one I am dealing with.

