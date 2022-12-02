"""
Handle server.

Classes:
    ServerThread

Author:
    Jose Angel de Angel
"""
#_________________________________Libraries____________________________________
from werkzeug.serving import make_server
import threading

#__________________________________Classes_____________________________________
class ServerThread(threading.Thread):
    """
    Server thread handler.
    
    ...
    
    Attributes
    ----------
    server : Server
        Server.
        
    ctx : App
        App.
    
    Methods
    -------
    run():
        Start server.
        
    shutdown():
        Stop server
    
    """
    
    def __init__(self, app):
        """
        Construct class attributes.

        Parameters
        ----------
        app : APP
            APP.

        Returns
        -------
        None.

        """
        threading.Thread.__init__(self)
        self.server = make_server('127.0.0.1', 7000, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        """
        Start server.

        Returns
        -------
        None.

        """        
        self.server.serve_forever()

    def shutdown(self):
        """
        Stop server.

        Returns
        -------
        None.

        """
        self.server.shutdown()