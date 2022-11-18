
# importing Flask and other modules
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.serving import make_server
import threading
import webbrowser
import time
import sys
import os

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def connect_to_network():
    if request.method == "POST":       
       network_name = request.form.get("nname")       
       password = request.form.get("password")
       #os.system("nmcli dev wifi connect {network_name} password {password}".format(network_name=network_name, password=password))
       #return redirect(url_for('initialization_form'))
       #os.system("^C")
              
    return render_template("Form1.html")
 
@app.route('/initialization_form', methods =["GET", "POST"])
def initialization_form():
    return render_template("Form2.html")

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

if __name__=='__main__':       
    #initialization_form_server = multiprocessing.Process(target= app.run(host = '0.0.0.0', port = 7000, debug = True) )       
    webbrowser.open("http://127.0.0.1:7000/")        
    #app.run(host = '0.0.0.0', port = 7000, debug = True)    
    global server
    server = ServerThread(app)
    server.start()
    print("st after server start")
    time.sleep(3)
    server.shutdown()


   
