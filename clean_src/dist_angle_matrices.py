import cPickle as pickle
import numpy as np

distances = np.zeros((240, 320))
angles = np.zeros((240, 320))
point = 320, 120

for y in range(240):
    for x in range(320):
        distances[y, x] = np.sqrt((x - .5 - point[0]) ** 2 +
                                  (y - .5 - point[1]) ** 2)
        angles[y, x] = (np.arctan2(-(y - .5 - point[1]), -(x - .5 - point[0]))
                        - np.pi / 2)

pickle.dump({"distances": distances, "angles": angles},
            open("distances.p", "wb"))
