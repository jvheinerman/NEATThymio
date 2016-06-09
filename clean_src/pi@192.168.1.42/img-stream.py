import BaseHTTPServer
import os

if not os.path.isdir("srv_resources"):
    os.system("install_server.py")

HOST_NAME = 'localhost' # !!!REMEMBER TO CHANGE THIS!!!
if os.getuid() == 0:
    PORT_NUMBER = 80
    print "hosting server on http://localhost"
else:
    PORT_NUMBER = 8080
    print "hosting server on http://localhost:8080"
try:
    css = open("srv_resources/bootstrap.min.css", "r").read()
    js = open("srv_resources/bootstrap.min.js", "r").read()
    bootstrap = '<style>'+css+'</style> <script type="text/javascript">'+js+'</script>'
    jquery = '<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>'
except:
    print "no bootstrap"

class ThymioStreamer(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

    def do_GET(s):
        """Respond to a GET request."""
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()
        s.wfile.write("<html><head><title>Thymio Live Stream</title>" +
                      jquery + bootstrap +"</head>")
        s.wfile.write("<body style='background-color: #000; color: #EEE'>")
        f = open("bots.txt")
        s.wfile.write("<div class='container-fluid'><div class='row'>")
        for bot in f:
            if bot.startswith("192"):
                s.wfile.write("<div class='col-xs-5'><h3>{0}</h3><img style='width: 100%; border: 5px solid #444' class='img-responsive' src='http://{0}:31337' />".format(bot[:-1]))
                s.wfile.write("<div class='col-xs-1'></div>")
        s.wfile.write("</div></div></body></html>")

if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), ThymioStreamer)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print
