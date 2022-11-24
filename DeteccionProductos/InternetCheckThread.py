import threading
import socket
import time

class InternetCheckThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_connected_to_internet = False
        self.is_running = False
        self.seconds_for_timeout = 10
        self.elapsed_time = 0    
        self.previous_time = None        

    def check_connection(self):
        self.is_running = True
        ip_address = "127.0.0.1"
        while ip_address == "127.0.0.1" and self.elapsed_time < self.seconds_for_timeout:
            ip_address=socket.gethostbyname(socket.gethostname())            
            self.elapsed_time += time.time() - self.previous_time
            self.previous_time = time.time()
        self.is_connected_to_internet = ip_address != "127.0.0.1"
        self.is_running = False        
        
    def run(self):
        self.previous_time = time.time()        
        self.check_connection()    

if __name__ == "__main__":
    i_thread = InternetCheckThread()
    i_thread.start()