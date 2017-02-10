#!/usr/bin/env python
import numpy as np
import pandas as pd
import pyproj
import gpxpy
import matplotlib.pyplot as plt
import sys
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from geojsonio import display
from geojson import dumps, Feature, Point, FeatureCollection, LineString


def cluster_by_k_means(frame, k=100):
    # Mean Normalize features
    scaler = preprocessing.StandardScaler()
    scaler.fit(frame[['x', 'y', 'altitude', 'time']])
    feature_columns = scaler.transform(frame[['x', 'y', 'altitude', 'time']])

    # Cluster
    print("Clustering k=", k, ") ...", file=sys.stderr, end="", flush=True)
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
        dt0 = frame.index[i].strftime('%d %b %Y %H:%M CUT')
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

                stops.append(
                    {'x': x, 'y': y, 'duration': duration, 'time': dt0})
            else:                               # from Started to Stopped
                # start capturing duration and stats
                t0 = row['time']
                dt0 = frame.index[i].strftime('%d %b %Y %H:%M CUT')
                i0 = i
            prev = row['stopped']

    # if track ended in Stopped state output the remaining items
    if prev:
        duration = (frame.iloc[i - 1]['time'] - frame.iloc[i0]['time'])
        x = frame.iloc[i0:i - 1]['x'].mean()
        y = frame.iloc[i0:i - 1]['y'].mean()
        stops.append({'x': x, 'y': y, 'duration': duration, 'time': dt0})

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
x_offset = {'raw': 0, 'resampled': 0}
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
    Feature(
        geometry=LineString(
            [tuple(x) for x in frame[['longitude', 'latitude']].values]
        )
    )
)
# gjs['resampled'].append(
#     LineString(
#         [tuple(x) for x in original[['longitude', 'latitude']].values]))

# original/resampled points, labelled o - moving or x - stopped
# gjs['raw'] = geojson_points_from_frame(
#     gjs['raw'], original, color['raw'])
# gjs['resampled'] = geojson_points_from_frame(
#     gjs['resampled'], frame, color['resampled'])


# we used this logic to present all potential stops for human assessment
counter = 1

for categ in ('raw', 'resampled'):
    print(categ)
    print("Number of stops:", len(stops[categ]), file=sys.stderr)
    print("Total duration of stops (hr):",
          sum(s['duration'] for s in stops[categ]) / 3600, file=sys.stderr)
    for s in stops[categ]:
        lon, lat = proj(s['x'] + x_offset[categ], s['y'], inverse=True)
        lon = round(lon, 5)
        lat = round(lat, 5)
        gjs['raw'].append(
            Feature(geometry=Point((lon, lat)),
                properties={
                    "duration": int(round(s['duration'] / 60, 0)),
                    "marker-color": color[categ],
                    "id": counter
                }
            )
        )
        s['id'] = counter
        s['lon'] = lon
        s['lat'] = lat
        counter = counter + 1
# print(dumps(FeatureCollection(gjs[categ])))
# display(dumps(FeatureCollection(gjs['raw'])), force_gist=True)

# These are the stops that were human-validated from above.
# Create a data set that has the locations and durations of validated stops
validated_stops = (1, 4, 5, 6, 7, 8, 9, 10, 12, 17, 18, 19, 21, 22, 23, 25,
                   26, 29, 30, 32, 33, 36, 37, 38, 40, 42, 44, 45, 46, 47,
                   51, 52, 54, 55, 58, 62, 64, 65, 67, 68, 70, 71, 72, 74,
                   76, 77, 78, 79, 82, 83, 85, 87, 88, 89, 90, 92, 99, 100,
                   101, 102, 103, 104, 106, 108, 110, 112, 117, 123, 126,
                   128, 130, 131, 132, 133, 135, 136, 138, 140, 142, 143,
                   145, 149, 150, 151, 153, 154, 158, 159, 160, 161, 163,
                   164, 167, 168, 174, 177, 179, 180, 183, 185, 186, 188,
                   192, 198, 203, 204, 206, 207, 211, 214, 215, 217, 219,
                   221, 222, 223, 225, 226, 227, 228, 229, 231, 233, 234,
                   236, 237, 239, 240, 243, 244, 245, 246, 248, 251, 254,
                   257, 258, 259, 260, 266, 267, 268, 270, 271, 272, 273,
                   275, 276, 277, 280, 282, 284, 285, 286, 288, 289, 290,
                   293, 294, 298, 299, 302, 303, 304, 305, 308, 310, 311,
                   313, 315, 318, 320, 321, 324, 326, 327, 328, 329, 333,
                   334, 335, 337, 343, 349, 350, 352, 363, 364, 365, 366,
                   367, 368, 369, 370, 376, 378, 379, 380, 381, 383, 384,
                   385, 386, 387, 388, 389, 390, 392, 393, 395, 396, 397,
                   400, 401, 402, 404, 406, 409, 411, 414, 418, 419, 421,
                   422, 423, 425, 427, 429, 430, 431, 433, 434, 435, 436,
                   437, 438, 439, 441, 442, 444, 445, 446, 450, 451, 453,
                   454, 458, 459, 460, 461, 462, 463, 467, 468, 471, 473,
                   475, 476, 480, 481, 482, 483, 484, 487, 545, 546, 628,
                   630, 638, 968)
gjs_stops = []
for categ in ('raw', 'resampled'):
    for s in stops[categ]:
        if s['id'] in validated_stops:
            gjs_stops.append(Feature(geometry=Point((s['lon'], s['lat'])),
                properties={
                    "point-type": "validated-stop",
                    "time": s['time'],
                    "duration": int(round(s['duration'] / 60, 0)),
                    "marker-color": '#ff0000',
                    "id": s['id']
                }))
display(dumps(FeatureCollection(gjs_stops)), force_gist=True)
with open('validated-stops.geojson', 'w') as outfile:
    json.dump(FeatureCollection(gjs_stops), outfile)

