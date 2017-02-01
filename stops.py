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

def cluster_by_k_means(frame, k=100):
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
    return centroids, labels


def cluster_by_interpoint_distance(frame, max_d=50, min_time=60, min_pts=3):
    i = 0
    stops = []
    duration = 0
    # Parse each point, and see if the following point is within max_d metres
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
            d2 = (
                frame.iloc[i]['x'] - x0) ** 2 + (frame.iloc[i]['y'] - y0) ** 2
            duration = (frame.iloc[i]['time'] - t0)

        # print("\n", file=sys.stderr, end="", flush=True)

        if (i - i0 > min_pts) and (duration > min_time):
            for t in range(i0, i):
                frame.set_value(frame.index[t], 'stopped', 1)

    # Loop once more through the list,
    # aggregating any adjacent stopped segments
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

    print("# stops:", len(stops), file=sys.stderr, end="\n", flush=True)
    return frame, stops


def geojson_points_from_frame(gjs, fr, color):
    for i in range(0, len(fr.index)):
        row = fr.iloc[i]
        gjs.append(
            Feature(
                geometry=Point((row['longitude'], row['latitude'])),
                properties={
                    "time": row['time'],
                    "marker-color": color,
                    "marker-symbol": \
                        "cross" if row['stopped'] else "circle"
                }
            )
        )
#    print("# features added:", len(fr), file=sys.stderr, end="\n", flush=True)
    return gjs


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

stops = {}
color = {'raw': '#ff0000', 'resampled': '#00ff00'}
x_offset = {'raw': 0, 'resampled': 100}
original, stops['raw'] = cluster_by_interpoint_distance(
    original, max_d, min_time, min_pts)
frame, stops['resampled'] = cluster_by_interpoint_distance(
    frame, max_d, min_time, min_pts)

plt.hist(
    [
        [s['duration'] for s in stops['raw']],
        [s['duration'] for s in stops['resampled']]
    ],
    bins=50,
    color=[color['raw'], color['resampled']],
    log=True
)

# stopped = frame.loc[frame['stopped'] == True]

# plt.scatter(x=original[['x']], y=original[['y']], marker='.')
# plt.scatter(x=stopped[['x']], y=stopped[['y']], marker='x')
# gjs_features = []
# gjs_features.append(
#     LineString([tuple(x) for x in frame[['longitude', 'latitude']].values])
#                    )
# for s in stops:
#     plt.annotate(
#         str(int(round(s['duration'] / 60, 0))),
#         (s['x'], s['y']))
#     lon, lat = proj(s['x'], s['y'], inverse=True)
#     lon = round(lon, 5)
#     lat = round(lat, 5)
#     gjs_features.append(
#         Feature(geometry=Point((lon, lat)),
#         properties={"duration": int(round(s['duration'] / 60, 0))}))

plt.show()
# display(dumps(FeatureCollection(gjs_features)), force_gist=True)

gjs = {}
gjs['raw'] = []
gjs['resampled'] = []

# original data points line
gjs['raw'].append(
    LineString(
        [tuple(x) for x in original[['longitude', 'latitude']].values]))
gjs['resampled'].append(
    LineString(
        [tuple(x) for x in original[['longitude', 'latitude']].values]))

# original/resampled points, labelled o - moving or x - stopped
# gjs['raw'] = geojson_points_from_frame(
#     gjs['raw'], original, color['raw'])
# gjs['resampled'] = geojson_points_from_frame(
#     gjs['resampled'], frame, color['resampled'])


# detected stops
for categ in ('raw', 'resampled'):
    print(categ)
    print("Number of stops:", len(stops[categ]), file=sys.stderr)
    print("Total duration of stops (hr):",
          sum(s['duration'] for s in stops[categ]) / 3600, file=sys.stderr)
    for s in stops[categ]:
        lon, lat = proj(s['x']+x_offset[categ], s['y'], inverse=True)
        lon = round(lon, 5)
        lat = round(lat, 5)
        gjs[categ].append(
            Feature(geometry=Point((lon, lat)),
                properties={
                    "duration": int(round(s['duration'] / 60, 0)),
                    "marker-color": color[categ]
                }
            )
        )
    display(dumps(FeatureCollection(gjs[categ])), force_gist=True)


