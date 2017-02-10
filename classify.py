import sys
import pandas as pd
import numpy as np
import json
import statistics
from datetime import datetime
from datetime import timedelta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_predict
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
from geojson import dumps, Feature, Point, FeatureCollection, LineString

infile = str(sys.argv[1])
dataset = pd.read_csv(infile)
interval = statistics.median([
    datetime.strptime(dataset['time'][i], '%Y-%m-%d %H:%M:%S').timestamp() -
    datetime.strptime(dataset['time'][i - 1], '%Y-%m-%d %H:%M:%S').timestamp()
    for i in range(1, len(dataset))
])


# columns = ["speed_prev", "rte_speed_prev", "dir_prev",
#            "speed_next", "rte_speed_next", "dir_next"]
columns = ["speed_prev", "dir_prev",
           "speed_next", "dir_next"]
labels = dataset['stopped'].values
features = dataset[list(columns)].values


# Normalizing
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)


# # Create range of classifiers
# et = ExtraTreesClassifier(
#     n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
# knn = KNeighborsClassifier()
# log = linear_model.LogisticRegression(C=1e5)
# svc = svm.SVC(kernel='linear')
# svc3 = svm.SVC(kernel='poly', degree=3)
# svc4 = svm.SVC(kernel='poly', degree=4)
# svcrbf = svm.SVC(kernel='rbf')
# lasso = linear_model.LassoCV()
# nn = MLPClassifier(
#     solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# gnb = GaussianNB()
# dt = tree.DecisionTreeClassifier()

# # Optimise the SVM using Grid Search
# Cs = np.logspace(-6, -1, 10)
# svcgrid = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)

# # Evaluate using a K-fold cross-validation
# for clf in [knn, log, svcrbf, svc3, svc4, svc, svcgrid, nn, gnb, dt]:
#     print(clf)
#     f1 = cross_val_score(clf, features, labels, cv=5, scoring='f1').mean()
#     print(f1, "\n")

# Train and classify using a Neural Network
nn = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
f1 = cross_val_score(nn, features, labels, cv=5, scoring='f1').mean()
print("NN cross-val F1 score:", f1)
predicted = cross_val_predict(nn, features, labels, cv=5)
dataset['stopped'] = predicted

# Loop once more through the list,
# aggregating any adjacent stopped segments
prev = False
stops = []
for i in range(0, len(dataset.index)):
    row = dataset.iloc[i]
    if (row['stopped'] != prev):   # change of state
        if prev:                            # from Stopped to Started
            # output previous run of Stops
            dt1 = dataset['time'][i - 1]
            t1 = datetime.strptime(dt1, '%Y-%m-%d %H:%M:%S').timestamp()
            duration = t1 - t0 + interval  ## add the median interval
            lat = dataset.iloc[i0:i]['latitude'].mean()
            lon = dataset.iloc[i0:i]['longitude'].mean()
            stops.append(
                {'lat': lat, 'lon': lon, 'duration': duration, 'time': dt0})
            # print(lat, lon, i0, i, dt0, "->", t0, dt1, "->", t1)
        else:                               # from Started to Stopped
            # start capturing duration and stats
            dt0 = dataset['time'][i]
            t0 = datetime.strptime(dt0, '%Y-%m-%d %H:%M:%S').timestamp()
            i0 = i
        prev = row['stopped']

# if track ended in Stopped state output the remaining items
if prev:
    dt1 = dataset['time'][i - 1]
    t1 = datetime.strptime(dt1, '%Y-%m-%d %H:%M:%S').timestamp()
    duration = t1 - t0 + interval
    lat = dataset.iloc[i0:i]['latitude'].mean()
    lon = dataset.iloc[i0:i]['longitude'].mean()
    stops.append({'lat': lat, 'lon': lon, 'duration': duration, 'time': dt0})
    # print(lat, lon, i0, i, dt0, "->", t0, dt1, "->", t1)

# print("# stops:", len(stops), file=sys.stderr, end="\n", flush=True)
gjs = []
for i in range(0, len(dataset)):
    p = dataset.iloc[i]
    if p['stopped']:
            gjs.append(
                Feature(
                    geometry=Point((p['longitude'], p['latitude'])),
                    properties={
                        "point-type": "predicted-stop",
                        "marker-size": "small",
                        "marker-symbol": "cross",
                        "marker-color": '#c00000'}
                )
            )
for s in stops:
    gjs.append(
        Feature(
            geometry=Point((s['lon'], s['lat'])),
            properties={
                "point-type": "predicted-stop-aggregated",
                "time": s['time'],
                "duration": str(timedelta(seconds=s['duration'])),
                "marker-color": '#ff0000'}
        )
    )
with open('nn-predictions.geojson', 'w') as outfile:
    json.dump(FeatureCollection(gjs), outfile)
