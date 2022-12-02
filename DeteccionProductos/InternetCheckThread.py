"""
Check internet connection.

Classes:
    InternetCheckThread
    
Author:
    Jose Angel del Angel

"""
#_________________________________Libraries____________________________________
import threading
import socket
import time

#__________________________________Classes_____________________________________
class InternetCheckThread(threading.Thread):
    """
    Class to heck internet connection.
    
    ...
    
    Attributes
    ----------
    is_connected_to_internet : bool
        True if connected to internet.
    
    is_running : bool
        True if server is running.
        
    seconds_for_timeout : int
        Seconds to wait before a timeout.
        
    elapsed_time : int
        Time elapsed.
    
    previous_time : int
        Previous elapsed time.
        
    Methods
    -------
    check_connection():
        Check internet connection.
        
    run():
        Check connection.
    
    """
    
    def __init__(self):
        """
        Construct class attributes.

        Returns
        -------
        None.

        """
        threading.Thread.__init__(self)
        self.is_connected_to_internet = False
        self.is_running = False
        self.seconds_for_timeout = 10
        self.elapsed_time = 0    
        self.previous_time = None        

    def check_connection(self):
        """
        Check internet connection.

        Returns
        -------
        None.

        """
        self.is_running = True
        ip_address = "127.0.0.1"
        while ip_address == "127.0.0.1" and self.elapsed_time < self.seconds_for_timeout:
            ip_address=socket.gethostbyname(socket.gethostname())            
            self.elapsed_time += time.time() - self.previous_time
            self.previous_time = time.time()
        self.is_connected_to_internet = ip_address != "127.0.0.1"
        self.is_running = False        
        
    def run(self):
        """
        Check internet connection and take time.

        Returns
        -------
        None.

        """
        self.previous_time = time.time()        
        self.check_connection()    

#____________________________________Main______________________________________
if __name__ == "__main__":
    i_thread = InternetCheckThread()
    i_thread.start()