#!/usr/bin/env python
import numpy as np
import pandas as pd
import pyproj
import gpxpy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from geojsonio import display
from geojson import dumps,  Feature, Point, FeatureCollection, LineString

def cluster_by_k_means (frame, k=100):
    # Mean Normalize features
    scaler = preprocessing.StandardScaler()
    scaler.fit(frame[['x', 'y', 'altitude', 'time']])
    feature_columns = scaler.transform(frame[['x', 'y', 'altitude', 'time']])

    # Cluster
    print("Clustering (k=", k, ") ...", file=sys.stderr, end="", flush=True)
    kmeans_model = KMeans(n_clusters=k, random_state=1)
    kmeans_model.fit(feature_columns)
    labels = kmeans_model.labels_
    centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)
    print("OK", len(centroids), "centroids found",
        file=sys.stderr, end="\n", flush=True)

    # print("Searching for stops...", file=sys.stderr, end="\n", flush=True)
    # for c in range(0, len(centroids)):
    #     # count how many points within 100m of the centroid
    #     cluster = frame.iloc[labels == c]
    #     d = cluster[['x', 'y']] - [centroids[c, 0], centroids[c, 1]]
    #     near = d[(abs(d['x']) < 100) & (abs(d['y']) < 100)]
    #     if len(near.index) > min_pts:
    #         lon, lat = proj(
    #             centroids[c, 0], centroids[c, 1], inverse=True)
    return centroids, labels


## MAIN ##
infile = str(sys.argv[1])

gpx = gpxpy.parse(open(infile))
seg_dict = {}
# TCR spans from Zone 31 to Zone 35. 33 is median. Hope it works!
proj = pyproj.Proj(proj='utm', zone=33, ellps='WGS84')

# Load the GPX file points into a Pandas data frame
print("Loading GPX file...", file=sys.stderr, end="", flush=True)
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            x, y = proj(point.longitude, point.latitude)
            stopped = False
            seg_dict[point.time] = [
                x, y,
                point.latitude, point.longitude, point.elevation, 0]
print("OK\n", file=sys.stderr, end="", flush=True)

print("Converting to DataFrame...", file=sys.stderr, end="", flush=True)
frame = pd.DataFrame(data=seg_dict)
# Switch columns and rows s.t. timestamps are rows and gps data columns.
frame = frame.T
frame.columns = ['x', 'y', 'latitude', 'longitude', 'altitude', 'stopped']
frame['time'] = (frame.index - frame.index[0]).astype(np.int64) // 10**9
print("OK\n", file=sys.stderr, end="", flush=True)

print("\nParameters of GPX track:", file=sys.stderr)
seconds = frame['time'].max() - frame['time'].min()
n = len(frame.index)
t_res = seconds / n
print("Number of points:", n, file=sys.stderr)
print("Duration (s):", seconds, file=sys.stderr)
print("Average resolution (s):", t_res, file=sys.stderr)
print("\nSelected processing parameters:", file=sys.stderr)
K = max(10, int(seconds / (60 * 30)))
if t_res < 20:
    freq = '10S'
elif t_res < 120:
    freq = 'min'
elif t_res < 600:
    freq = '5min'
else:
    freq = '10min'
min_pts = 3
min_time = 60
max_d = 50
print("Resampled frequency:", freq, file=sys.stderr)
print("Number of clusters (K):", K, file=sys.stderr)
print("Minimum points to detect a stop:", min_pts, file=sys.stderr)
print("Minimum time (s) to detect a stop:", min_time, file=sys.stderr)
print("Maximum travel (m) within a stop:", max_d, file=sys.stderr)

# Resample / fill / interpolate samples
print("Resampling...", file=sys.stderr, end="", flush=True)
original = frame
frame = frame.resample(freq).mean().interpolate(method='linear')
print("OK\n", file=sys.stderr, end="", flush=True)

# print("Reconverting lat/lon...", file=sys.stderr, end="", flush=True)
# for i in range(0, frame.index.size):
#     p = frame.iloc[i]
#     # lon/lat can't be safely interpolated so refresh them here
#     p.longitude, p.latitude = proj(p.x, p.y, inverse=True)
# print("OK\n", file=sys.stderr, end="", flush=True)

# Let's make features: distance^2 (dX^2 + dY^2) at +1, +2, +3, +4 min
# all_features = {}
# n = 4

# for i in range(0, frame.index.size):
#     features = []
#     for f in range(1, n + 1):
#         if (i + f < frame.index.size):
#             features.append(
#                 (frame.iloc[i + f].x - frame.iloc[i].x) ** 2 + (frame.iloc[i + f].y - frame.iloc[i].y) ** 2
#             )
#             # features.append(frame.iloc[i + f].x - frame.iloc[i].x)
#             # features.append(frame.iloc[i + f].y - frame.iloc[i].y)
#         else:           # zero-fill the last few slots
#             features.append(0)
#     all_features[frame.index[i]] = features

# frame = pd.DataFrame(data=all_features)
# frame = frame.T
# frame.columns = ['d1', 'd2', 'd3', 'd4']

# frame = frame.join(frame)



# Cluster using KMeans
# centroids, labels = cluster_by_k_means(frame, K)



i = 0
stops = []
duration = 0
# Run through each point, and see if the following point is within max_d metres
# If so, and other minimum conditions are met, label it as Stopped
while (i < len(frame.index) - 1):
    t0 = frame.iloc[i]['time']
    lat0 = frame.iloc[i]['latitude']
    lon0 = frame.iloc[i]['longitude']
    x0 = frame.iloc[i]['x']
    y0 = frame.iloc[i]['y']
    i0 = i
    d2 = 0

    while (d2 < max_d ** 2) and (i < len(frame.index) - 1):
        # print(".", file=sys.stderr, end="", flush=True)
        i = i + 1
        d2 = (frame.iloc[i]['x'] - x0) ** 2 + (frame.iloc[i]['y'] - y0) ** 2
        duration = (frame.iloc[i]['time'] - t0)

    # print("\n", file=sys.stderr, end="", flush=True)

    if (i - i0 > min_pts) and (duration > min_time):
        for t in range(i0, i):
            frame.set_value(frame.index[t], 'stopped', 1)

# Loop once more through the list, aggregating any adjacent stopped segments
prev = False
for i in range(0, len(frame.index)):
    row = frame.iloc[i]
    if (row['stopped'] != prev):   # change of state
        if prev:                            # from Stopped to Started
            # output previous run of Stops
            duration = (frame.iloc[i - 1]['time'] - t0)
            x = frame.iloc[i0:i - 1]['x'].mean()
            y = frame.iloc[i0:i - 1]['y'].mean()

            stops.append({'x': x, 'y': y, 'duration': duration})
        else:                               # from Started to Stopped
            # start capturing duration and stats
            t0 = row['time']
            i0 = i
        prev = row['stopped']

# if track ended in Stopped state output the remaining items
if prev:
    duration = (frame.iloc[i - 1]['time'] - frame.iloc[i0]['time'])
    x = frame.iloc[i0:i - 1]['x'].mean()
    y = frame.iloc[i0:i - 1]['y'].mean()
    stops.append({'x': x, 'y': y, 'duration': duration})

stopped = frame.loc[frame['stopped'] == True]

plt.scatter(x=original[['x']], y=original[['y']], marker='.')
plt.scatter(x=stopped[['x']], y=stopped[['y']], marker='x')
gjs_features = []
gjs_features.append(
    LineString([tuple(x) for x in frame[['longitude', 'latitude']].values])
                   )
for s in stops:
    plt.annotate(
        str(int(round(s['duration'] / 60, 0))),
        (s['x'], s['y']))
    lon, lat = proj(s['x'], s['y'], inverse=True)
    lon = round(lon, 5)
    lat = round(lat, 5)
    gjs_features.append(
        Feature(geometry=Point((lon, lat)),
        properties={"duration": int(round(s['duration'] / 60, 0))}))

plt.show()
display(dumps(FeatureCollection(gjs_features)), force_gist=True)



print("Number of stops:", len(stops), file=sys.stderr)
print("Total duration of stops (hr):",
      sum(s['duration'] for s in stops) / 3600, file=sys.stderr)

