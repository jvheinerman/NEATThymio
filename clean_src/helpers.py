import sys, os, errno
from parameters import MAX_MOTOR_SPEED

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

def getNextIDPath(path):
    nextID = 0
    filelist = sorted(os.listdir(path))
    if filelist and filelist[-1][0].isdigit():
        nextID = int(filelist[-1][0]) + 1
    return str(nextID)

def writeMotorSpeed(controller, motorspeed, max_speed=MAX_MOTOR_SPEED):
    controller.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * max_speed])
    controller.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * max_speed])


def getProxReadings(controller, ok_callback, nok_callback):
    controller.GetVariable("thymio-II", "prox.horizontal", reply_handler=ok_callback, error_handler=nok_callback)

def stopThymio(controller):
    writeMotorSpeed(controller, { 'left': 0, 'right': 0 })

def dbusReply():
    pass


def dbusError(e):
    print 'error %s' % str(e)
