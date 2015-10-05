import sys

RAND_MAX = sys.maxint
LEFT = 0
RIGHT = 1

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Uniform distribution (0..1]
def drand():
    return random.randint(0, RAND_MAX) / float(RAND_MAX + 1)

# Normal distribution, centered on 0, std dev 1
def random_normal():
    return -2 * math.log(drand())

# Used because else webots gives a strange segfault during cross compilation
def sqrt_rand_normal():
    return math.sqrt(random_normal())

def gaussrand():
    return sqrt_rand_normal() * math.cos(2 * math.pi * drand())