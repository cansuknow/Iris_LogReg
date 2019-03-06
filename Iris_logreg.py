import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

logreg = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial')

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

logreg.predict_proba(X_test)

metrics.accuracy_score(y_test,y_pred)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

iris2 = load_iris()
X2 = iris2.data[:, 2:4]
y2 = iris2.target

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=4)

logreg2 = LogisticRegression(C=0.1, solver='lbfgs', multi_class='multinomial')

logreg2.fit(X_train2, y_train2)

y_pred2 = logreg2.predict(X_test2)

logreg2.predict_proba(X_test2)

metrics.accuracy_score(y_test2,y_pred2)

x_min2, x_max2 = X2[:, 0].min() - .5, X2[:, 0].max() + .5
y_min2, y_max2 = X2[:, 1].min() - .5, X2[:, 1].max() + .5
h = .02
xx2, yy2 = np.meshgrid(np.arange(x_min2, x_max2, h), np.arange(y_min2, y_max2, h))

Z2 = logreg2.predict(np.c_[xx2.ravel(), yy2.ravel()])
Z2 = Z2.reshape(xx2.shape)

plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx2, yy2, Z2, cmap=plt.cm.Paired)

plt.scatter(X2[:, 0], X2[:, 1], c=y2, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.xlim(xx2.min(), xx2.max())
plt.ylim(yy2.min(), yy2.max())
plt.xticks(())
plt.yticks(())

plt.show()

