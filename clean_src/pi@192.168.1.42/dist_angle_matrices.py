import cPickle as pickle
import numpy as np

distances = np.zeros((240, 320))
angles = np.zeros((240, 320))
point = 240, 160

for x in range(240):
    for y in range(320):
        distances[x, y] = np.sqrt((x - .5 - point[0]) ** 2 +
                                  (y - .5 - point[1]) ** 2)
        angles[x, y] = -(np.arctan2(-(y - .5 - point[1]), -(x - .5 - point[0]))
                         - np.pi / 2) - np.pi / 2

pickle.dump({"distances": distances, "angles": angles},
            open("distances.p", "wb"))
