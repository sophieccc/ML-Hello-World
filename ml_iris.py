from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from math import floor
from sklearn.tree import tree
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state=0)
predictions = {}

# logistic regression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)
predictions["Logistic Regression"] = predict

# decision trees

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)
predictions["Decision Tree"] = predict

# k neighbours

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)
predictions["K Neighbours"] = predict

# results

for key in predictions : 
    score = accuracy_score(predictions.get(key), y_test)*100
    print("Accuracy score of {:.2f}%.\n".format(score))
    print(classification_report(y_test, predictions.get(key)))
