from werkzeug.serving import make_server
import threading

class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server('127.0.0.1', 7000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):        
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()