import os, threading, socket, select, struct, pickle, errno, shutil, time

N_THYMIOS = 6
OUTPUT_FILE_RECEIVER_PORT = 23456
CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
RECEIVED_OUTPUTS_PATH = os.path.join(CURRENT_FILE_PATH, 'received_outputs')
ALGORITHM_PATH = os.path.join(os.path.dirname(CURRENT_FILE_PATH), 'PycharmProjects', 'test')
EOF_REACHED = 'EOF_REACHED'

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def recvall(conn, count):
    buf = b''
    while count:
        newbuf = conn.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def recvOneMessage(socket):
    lengthbuf = recvall(socket, 4)
    if not lengthbuf: return None
    length, = struct.unpack('!I', lengthbuf)
    recvD = recvall(socket, length)
    data = pickle.loads(recvD)
    return data

def sendOneMessage(conn, data):
    packed_data = pickle.dumps(data)
    length = len(packed_data)
    conn.sendall(struct.pack('!I', length))
    conn.sendall(packed_data)

class RecvFileThread(threading.Thread):
    def __init__(self, conn, addr):
        threading.Thread.__init__(self)
        self.__conn = conn
        self.__addr = addr
        self.savingDir = None

    def run(self):
        print str(self.__addr) + " Receiving experimentName..."
        experimentName = recvOneMessage(self.__conn)
        self.savingDir = os.path.join(RECEIVED_OUTPUTS_PATH, experimentName)
        mkdir_p(self.savingDir)
        
        # Receive output file
        print str(self.__addr) + " Receiving output..."
        fO = open(self.savingDir + '/' + experimentName + '_' + str(self.__addr) + '_out.txt', 'wb')
        l = recvOneMessage(self.__conn)
        while l:
            fO.write(l)
            l = recvOneMessage(self.__conn)
            if l == EOF_REACHED:
                break
        fO.close()
        
        print str(self.__addr) + " Receiving log..."
        fL = open(self.savingDir + '/' + experimentName + '_' + str(self.__addr) + '_sim_debug.log', 'wb')
        l = recvOneMessage(self.__conn)
        while l:
            fL.write(l)
            l = recvOneMessage(self.__conn)
            if l == EOF_REACHED:
                break
        fL.close()

        print str(self.__addr) + " Receiving temperatue..."
        fT = open(self.savingDir + '/' + experimentName + '_' + str(self.__addr) + '_temp.txt', 'wb')
        l = recvOneMessage(self.__conn)
        while l:
            fT.write(l)
            l = recvOneMessage(self.__conn)
            if l == EOF_REACHED:
                break
        fT.close()

        print str(self.__addr) + " Receiving weight..."
        fW = open(self.savingDir + '/' + experimentName + '_' + str(self.__addr) + '_weight_out.txt', 'wb')
        l = recvOneMessage(self.__conn)
        while l:
            fW.write(l)
            l = recvOneMessage(self.__conn)
        fW.close()

if __name__ == '__main__':
    print "START: " + time.strftime("%Y-%m-%d_%H-%M-%S")
    s = socket.socket()         # Create a socket object
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', OUTPUT_FILE_RECEIVER_PORT))        # Bind to the port
    s.listen(N_THYMIOS)                 # Now wait for client connection.
    recvThreads = list()
    savingDir = None

    for i in range(0, N_THYMIOS):
        c, (addr, port) = s.accept()     # Establish connection with client.
        t = RecvFileThread(c, addr)
        recvThreads.append(t)
        t.start()
    for rT in recvThreads:
        rT.join()
        savingDir = rT.savingDir

    s.close()
    shutil.copyfile(ALGORITHM_PATH + "/algorithmForaging1NN.py", savingDir + "/algorithmForaging1NN.py")
    shutil.copyfile(ALGORITHM_PATH + "/parameters1NN.py", savingDir + "/parameters1NN.py")
    print "END: " + time.strftime("%Y-%m-%d_%H-%M-%S")