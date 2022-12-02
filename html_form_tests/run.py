"""
Test for local server and html use.

Classes:
    ServerThread
    
Functions:
    connect_to_network() -> Rendered Template
    
    initialization_form() -> Rendered Template

Author:
    Jose Angel del Angel
    
"""
#_________________________________Libraries____________________________________
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.serving import make_server
import threading
import webbrowser
import time

#_________________________________Variables____________________________________
# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])

#_________________________________Functions____________________________________
def connect_to_network():
    """
    Connect to a network.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    Rendered Template
        First form.

    """
    # Connect
    if request.method == "POST":       
        network_name = request.form.get("nname")       
        password = request.form.get("password")
        #os.system("nmcli dev wifi connect {network_name} password {password}".format(network_name=network_name, password=password))
        if network_name != "" and password != "":
            return redirect(url_for('initialization_form'))                     
    return render_template("Form1.html")
 
@app.route('/initialization_form', methods =["GET", "POST"])
def initialization_form():
    """
    Builds initialization form.

    Parameters
    ----------
    None.

    Returns
    -------
    Rendered Template
        Rendered form 2.

    """
    product_block = []
    for i in range(2):
        product_name = 'Fanta'
        product_ean_current_stock="\"6848564646\"_current"
        product_ean_min_stock="\"6848564646\"_min"
        product_ean_max_stock="\"6848564646\"_max"
        product_html_code = {"name": product_name , "current": product_ean_current_stock, "min": product_ean_min_stock, "max": product_ean_max_stock}
        product_block.append(product_html_code)

    return render_template("Form2.html", product_block = product_block)

class ServerThread(threading.Thread):
    """
    Class that runs server.
    
    Attributes
    ----------
    server : Server
        Server.
        
    ctx : App Context
        App Context.
        
    Methods
    -------
    run():
        Start server.
    
    shutdown():
        Stop server.
        
    """
    
    def __init__(self, app):
        """
        Construct attributes.

        Parameters
        ----------
        app : App
            Flask app.

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
        Initialize sever.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        self.server.serve_forever()

    def shutdown(self):
        """
        Stop server.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.

        """
        self.server.shutdown()

#____________________________________Main______________________________________
if __name__=='__main__':       
    #initialization_form_server = multiprocessing.Process(target= app.run(host = '0.0.0.0', port = 7000, debug = True) )                 
    #app.run(host = '0.0.0.0', port = 7000, debug = True)    
    global server
    server = ServerThread(app)
    server.start()
    webbrowser.open("http://127.0.0.1:7000/initialization_form")  
    print("st after server start")
    time.sleep(3)
    server.shutdown()


   
