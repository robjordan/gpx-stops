import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

dataset = pd.read_csv('61-with-features-and-labels.csv')

et = ExtraTreesClassifier(
    n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
columns = ["speed_prev", "dir_prev", "speed_next", "dir_next"]
labels = dataset['stopped'].values
features = dataset[list(columns)].values

# Normalizing
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)


# Create range of classifiers
knn = KNeighborsClassifier()
log = linear_model.LogisticRegression(C=1e5)
svc = svm.SVC(kernel='linear')
svc3 = svm.SVC(kernel='poly', degree=3)
svc4 = svm.SVC(kernel='poly', degree=4)
svcrbf = svm.SVC(kernel='rbf')
lasso = linear_model.LassoCV()
nn = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
gnb = GaussianNB()
dt = tree.DecisionTreeClassifier()

# Optimise the SVM using Grid Search
Cs = np.logspace(-6, -1, 10)
svcgrid = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)

# Evaluate using a K-fold cross-validation
for clf in [knn, log, svcrbf, svc3, svc4, svc, svcgrid, nn, gnb, dt]:
    print(clf)
    f1 = cross_val_score(clf, features, labels, cv=5, scoring='f1').mean()
    print(f1, "\n")

