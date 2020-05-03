from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from math import floor

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)

score = accuracy_score(predict, y_test)*100
print("Accuracy score of {:.2f}%.\n".format(score))
print(classification_report(y_test, predict))