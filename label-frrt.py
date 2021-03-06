#!/usr/bin/env python
import numpy as np
import pandas as pd
import pyproj
from pyproj import Geod
import gpxpy
import matplotlib.pyplot as plt
import sys
import math
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from geojsonio import display
from geojson import dumps, Feature, Point, FeatureCollection, LineString, Polygon

stop_coords = [(3.86754, 50.64401),
     (3.68104, 50.16642),
     (3.61679, 49.77032),
     (3.63221, 49.56857),
     (3.55377, 49.47112),
     (3.4658, 49.36155),
     (3.47665, 49.19358),
     (3.42124, 49.16079),
     (3.41982, 49.15967),
     (3.32828, 48.84093),
     (3.31632, 48.80512),
     (3.30248, 48.78182),
     (3.30283, 48.7821),
     (3.3085, 48.78428),
     (3.15124, 48.55738),
     (3.16704, 48.43962),
     (3.12226, 48.31245),
     (3.14458, 47.96835),
     (3.09505, 47.88668),
     (3.25564, 47.46296),
     (3.22066, 47.19196),
     (3.1575, 46.98834),
     (3.35242, 46.71581),
     (3.35276, 46.70002),
     (3.3009, 46.38152),
     (3.26561, 46.13176),
     (3.11825, 45.79206),
     (3.10936, 45.78921),
     (3.01061, 45.76951),
     (2.95568, 45.76395),
     (3.09892, 45.83039),
     (3.40138, 46.10739),
     (3.42496, 46.12546),
     (3.42486, 46.12496),
     (4.06923, 46.35387),
     (4.95168, 46.66869),
     (5.03693, 46.69329),
     (5.24307, 46.7526),
     (5.45262, 46.74582),
     (5.60419, 46.76967),
     (5.84576, 46.78998),
     (5.89543, 46.78004),
     (5.94016, 46.77803),
     (6.28352, 46.78391),
     (6.28541, 46.77949),
     (6.38892, 46.75998),
     (6.38942, 46.73519),
     (6.54527, 46.77399),
     (6.64048, 46.77987),
     (6.6418, 46.77911),
     (6.7966, 46.82148),
     (7.2808, 46.92217),
     (7.39531, 46.94203),
     (7.4638, 46.95852),
     (7.52257, 46.91916),
     (7.62157, 46.72393),
     (8.02894, 46.62523),
     (8.0312, 46.62447),
     (8.03511, 46.62423),
     (8.04077, 46.62371),
     (8.04157, 46.62406),
     (8.04183, 46.62407),
     (8.04632, 46.62524),
     (8.07452, 46.63635),
     (8.08213, 46.64209),
     (8.08701, 46.64567),
     (8.10208, 46.65578),
     (8.21147, 46.71435),
     (8.30562, 46.61492),
     (8.33781, 46.5737),
     (8.33867, 46.56587),
     (8.35938, 46.56768),
     (8.38846, 46.57662),
     (8.41513, 46.5727),
     (8.74432, 46.66906),
     (9.19187, 46.77477),
     (9.29665, 46.78867),
     (9.34508, 46.80036),
     (9.4014, 46.81205),
     (9.40066, 46.81187),
     (9.50206, 46.68765),
     (9.54382, 46.67792),
     (9.54564, 46.67712),
     (9.68753, 46.67133),
     (9.7453, 46.62899),
     (9.76037, 46.59428),
     (9.83764, 46.58232),
     (9.91995, 46.58044),
     (9.92555, 46.57893),
     (9.95939, 46.60192),
     (10.09248, 46.69958),
     (10.149, 46.68501),
     (10.29253, 46.63955),
     (10.56251, 46.67174),
     (11.08477, 46.67722),
     (11.29954, 46.48498),
     (11.2958, 46.34498),
     (11.30519, 46.33245),
     (11.59799, 46.30598),
     (11.69855, 46.36628),
     (11.7912, 46.37836),
     (12.01993, 46.40643),
     (12.03519, 46.44987),
     (12.03599, 46.47514),
     (12.05338, 46.48263),
     (12.05977, 46.44555),
     (12.10274, 46.39302),
     (12.2493, 46.2757),
     (12.28714, 46.19383),
     (12.31152, 45.97044),
     (12.47615, 45.94774),
     (12.91121, 45.96446),
     (13.53336, 45.8098),
     (13.57783, 45.79849),
     (13.58996, 45.78896),
     (13.79485, 45.6856),
     (13.83961, 45.65694),
     (13.86297, 45.64343),
     (13.86286, 45.64153),
     (14.01741, 45.56925),
     (14.22621, 45.48896),
     (14.25178, 45.48124),
     (14.27806, 45.47313),
     (14.37804, 45.34708),
     (14.43289, 45.33172),
     (14.44504, 45.3252),
     (14.44516, 45.32458),
     (14.49305, 45.32482),
     (14.57521, 45.29818),
     (14.57845, 45.29669),
     (14.64955, 45.24075),
     (14.69852, 45.21553),
     (14.70518, 45.16652),
     (14.79503, 45.12497),
     (14.79143, 45.12571),
     (14.78647, 45.1286),
     (14.78833, 45.12558),
     (14.79062, 45.12801),
     (14.90037, 44.99261),
     (14.90326, 44.99127),
     (14.90348, 44.99123),
     (14.97128, 44.97233),
     (14.975, 44.98269),
     (15.07836, 44.96428),
     (15.18416, 44.89355),
     (15.23401, 44.8686),
     (15.23583, 44.86971),
     (15.23385, 44.86861),
     (15.37135, 44.70849),
     (15.51479, 44.44417),
     (15.91383, 44.27422),
     (15.94693, 44.27057),
     (16.08178, 44.09447),
     (16.19764, 44.04245),
     (16.20817, 44.04076),
     (16.21773, 44.04363),
     (16.26599, 44.04037),
     (16.3084, 44.02753),
     (16.35664, 43.97916),
     (16.39549, 43.93565),
     (16.63793, 43.72932),
     (16.64049, 43.70935),
     (16.64071, 43.6991),
     (16.6374, 43.69694),
     (16.87675, 43.53199),
     (17.08879, 43.48022),
     (17.19586, 43.44363),
     (17.27704, 43.42169),
     (17.41591, 43.37143),
     (17.52998, 43.20933),
     (17.70558, 43.11433),
     (17.77387, 43.13406),
     (18.03497, 43.07263),
     (18.13648, 43.0887),
     (18.26905, 43.05344),
     (18.35419, 43.01456),
     (18.40155, 42.97131),
     (18.41572, 42.97488),
     (18.4198, 42.97888),
     (18.4269, 43.01155),
     (18.58892, 43.13722),
     (18.62872, 43.15694),
     (18.65226, 43.15711),
     (18.66148, 43.15989),
     (18.84151, 43.15342),
     (19.02183, 43.10495),
     (19.05536, 43.10011),
     (19.12771, 43.15426),
     (19.12086, 43.15552),
     (19.29216, 43.14869),
     (19.29387, 43.14925),
     (19.29634, 43.16327),
     (19.30878, 43.16674),
     (19.42998, 43.22142),
     (19.61567, 43.1659),
     (19.85952, 42.91327),
     (19.92594, 42.83941),
     (19.96371, 42.83586),
     (20.00098, 42.83433),
     (20.14591, 42.83281),
     (20.14883, 42.83377),
     (20.1445, 42.83115),
     (20.13769, 42.8316),
     (20.15422, 42.83566),
     (20.17997, 42.80152),
     (20.22092, 42.80038),
     (20.28892, 42.75886),
     (20.32054, 42.73332),
     (20.56963, 42.62947),
     (20.56968, 42.62928),
     (20.57369, 42.6241),
     (20.58429, 42.59124),
     (20.81259, 42.45319),
     (20.84555, 42.4176),
     (21.04007, 42.43247),
     (21.26501, 42.17731),
     (21.30465, 42.13863),
     (21.37647, 42.05954),
     (21.73549, 41.82214),
     (21.73551, 41.8223),
     (21.75517, 41.77837),
     (21.78427, 41.71601),
     (21.89826, 41.62488),
     (21.94954, 41.57506),
     (21.98106, 41.55448),
     (22.09974, 41.49931),
     (22.42957, 41.34491),
     (22.44546, 41.33996),
     (22.5654, 41.3171),
     (22.74872, 41.17095),
     (22.75885, 41.17715),
     (22.7962, 41.25364),
     (22.84277, 41.26366),
     (22.96958, 41.28376),
     (23.20283, 41.26213),
     (23.31161, 41.21288),
     (23.47338, 41.09575),
     (23.54725, 41.08984),
     (23.92493, 41.0111),
     (23.98322, 40.99465),
     (23.99357, 40.98862),
     (24.3204, 40.99163),
     (24.38855, 40.94171),
     (24.411, 40.93632),
     (24.42402, 40.94399),
     (24.89517, 41.1317),
     (24.89645, 41.14083),
     (24.89632, 41.14094),
     (25.02442, 41.12946),
     (25.18047, 41.12631),
     (25.28869, 41.12712),
     (25.64504, 40.96258),
     (25.88878, 40.84823),
     (26.20526, 40.9283),
     (26.31109, 40.94497),
     (26.32967, 40.9333),
     (26.33246, 40.93077),
     (26.52005, 40.87574),
     (26.6235, 40.86547),
     (26.62488, 40.86675),
     (26.77076, 40.70795),
     (26.82689, 40.65463),
     (26.64406, 40.40653),
     (4.27475, 46.434),
     (4.27421, 46.43447),
     (8.56794, 46.61792),
     (8.59617, 46.63498),
     (8.67119, 46.65966),
     (24.33023, 40.98429),
     (12.0348,46.4514),
     (14.80375, 45.12373)]

proximity = 75 ** 2


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def speed_xy(x1, y1, t1, x2, y2, t2):
    return distance(x1, y1, x2, y2) / float((t2 - t1).total_seconds())


def speed_ll(geod, lat1, lon1, t1, lat2, lon2, t2):
    az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)
    return dist / float((t2 - t1).total_seconds())


def direction_xy(x1, y1, x2, y2):
    return math.atan2((y2 - y1), (x2 - x1))


def load_gpx_to_frame(file):
    gpx = gpxpy.parse(open(file))
    seg_dict = {}
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                x, y = proj(point.longitude, point.latitude)
                stopped = False
                seg_dict[point.time] = [
                    x, y,
                    point.latitude, point.longitude, point.elevation, 0]
    frame = pd.DataFrame(data=seg_dict)
    # Switch columns and rows s.t. timestamps are rows and gps data columns.
    frame = frame.T
    frame.columns = ['x', 'y', 'latitude', 'longitude', 'altitude', 'stopped']
    return frame


def xy_rte_pos_at_time_t(rte, t):
    idx = np.searchsorted(rte.index.values, np.datetime64(t))
    return idx


def rte_distance_and_speed(rte, x1, y1, t1, x2, y2, t2):
    d = 0.0
    idx1 = xy_rte_pos_at_time_t(rte, t1)
    idx2 = xy_rte_pos_at_time_t(rte, t2)
    # first the bit from x1, y1 to x, y @ idx1
    d = d + distance(x1, y1, rte.iloc[idx1]['x'], rte.iloc[idx1]['y'])
    # then all the hops from idx1 to (idx2-1)
    for i in range(idx1, idx2 - 1):
        d = d + distance(
            rte.iloc[i]['x'],
            rte.iloc[i]['y'],
            rte.iloc[i + 1]['x'],
            rte.iloc[i + 1]['y'])
    # finally the bit from (idx2-1) to x2, y2
    d = d + distance(rte.iloc[idx2 - 1]['x'], rte.iloc[idx2 - 1]['y'], x2, y2)

    t1 = np.datetime64(t1).astype('uint64')
    t2 = np.datetime64(t2).astype('uint64')
    speed = d / ((t2 - t1) // 1E6)
    return d, speed

## MAIN ##


# TCR spans from Zone 31 to Zone 35. 33 is median. Hope it works!
proj = pyproj.Proj(proj='utm', zone=33, ellps='WGS84')
wgs84 = Geod(ellps='WGS84')

# Load the GPX file points into a Pandas data frame
print("Loading GPX track file...", file=sys.stderr, end="", flush=True)
trk_file = str(sys.argv[1])
frame = load_gpx_to_frame(trk_file)
print("OK\n", file=sys.stderr, end="", flush=True)
print("Loading GPX route file...", file=sys.stderr, end="", flush=True)
rte_file = str(sys.argv[2])
rte_frame = load_gpx_to_frame(rte_file)
print("OK\n", file=sys.stderr, end="", flush=True)


frame.columns = ['x', 'y', 'latitude', 'longitude', 'altitude', 'stopped']
frame['time'] = frame.index
frame['speed_prev'] = 0.0
frame['rte_speed_prev'] = 0.0
frame['dir_prev'] = 0.0
frame['speed_next'] = 0.0
frame['rte_speed_next'] = 0.0
frame['rte_dist_next'] = 0.0
frame['dir_next'] = 0.0
frame = frame.reset_index(drop=True)
print("OK\n", file=sys.stderr, end="", flush=True)

# Brute force distance to the array of validated stops
print("Searching for neighbours...", file=sys.stderr, end="", flush=True)
stops = []
for lon, lat in stop_coords:
    x, y = proj(lon, lat)
    stops.append((x, y))

for i in range(0, len(frame.index)):
    row = frame.iloc[i]
    x1 = row['x']
    y1 = row['y']
    t = row['time']
    for x2, y2 in stops:
        if (((x1 - x2) ** 2 + (y1 - y2) ** 2) < proximity):
            # print("stopped (", x1, y1, ") (", x2, y2, ")",
            #     file=sys.stderr, end="\n", flush=True)
            frame.set_value(frame.index[i], 'stopped', 1)
    # Add some features
    if i > 0:
        prev = frame.iloc[i - 1]
        frame.set_value(
            frame.index[i],
            'speed_prev',
            speed_xy(prev['x'], prev['y'], prev['time'], x1, y1, t))
        frame.set_value(
            frame.index[i],
            'dir_prev',
            direction_xy(prev['x'], prev['y'], x1, y1))
        dist, speed = rte_distance_and_speed(
            rte_frame,
            prev['x'],
            prev['y'],
            prev['time'],
            x1,
            y1,
            t)
        frame.set_value(
            frame.index[i],
            'rte_speed_prev',
            speed)
    if i < len(frame.index) - 1:
        next = frame.iloc[i + 1]
        frame.set_value(
            frame.index[i],
            'speed_next',
            speed_xy(x1, y1, t, next['x'], next['y'], next['time']))
        frame.set_value(
            frame.index[i],
            'dir_next',
            direction_xy(x1, y1, next['x'], next['y']))
        dist, speed = rte_distance_and_speed(
            rte_frame,
            x1,
            y1,
            t,
            next['x'],
            next['y'],
            next['time'])
        frame.set_value(
            frame.index[i],
            'rte_speed_next',
            speed)
        frame.set_value(
            frame.index[i],
            'rte_dist_next',
            dist)
# Write the features and labels to a CSV file for later classification work
frame.to_csv(trk_file + ".csv")

# Visualise
gjs = []
for sx, sy in stops:
    gjs.append(Feature(
        geometry=Polygon([[
            proj(sx - 75, sy, inverse=True),
            proj(sx, sy + 75, inverse=True),
            proj(sx + 75, sy, inverse=True),
            proj(sx, sy - 75, inverse=True)]]
        ),
        properties={"fill": '#ff0000'}
    ))
with open('validated-stop-polygons.geojson', 'w') as outfile:
    json.dump(FeatureCollection(gjs), outfile)
for i in range(0, len(frame.index)):
    row = frame.iloc[i]
    if (row['stopped']):
        color = '#ff0000'
    else:
        color = '#00ff00'
    gjs.append(
        Feature(
            geometry=Point((row['longitude'], row['latitude'])),
            properties={"marker-color": color}
        )
    )


# display(dumps(FeatureCollection(gjs)), force_gist=True)

