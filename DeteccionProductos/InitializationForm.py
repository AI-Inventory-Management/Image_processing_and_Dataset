"""
Initialization form.

    Render form templates and process information.
    
Functions:
    get_stock_tag_names()
    
    connect_to_network() -> Rendered Template
    
    initialization_form() -> Rendered Template
    
    success_page() -> Rendered Template
    
    connecting() -> Rendered Template
    
    check_internet_thread_status() -> URL
    
    load_form(dict, bool) -> Server Thread
    
Author:
    Jose Angel del Angel
"""
#_________________________________Libraries____________________________________
from flask import Flask, request, render_template, redirect, url_for
import ServerThread
import InternetCheckThread
import webbrowser
import time
import os

#_________________________________Variables____________________________________
app = Flask(__name__)
server = ServerThread.ServerThread(app)
internet_check_thread = None
form_complete = False
running_on_intel_nuc = False
eans2labels = None
stock_tag_names = []
form_data = {
    "store_name": None,
    "store_latitude": None,
    "store_longitude": None,
    "store_state": None,
    "store_municipality": None,
    "store_zip_code": None,
    "store_address": None,
    "current_stock": {},        
    "min_stocks": {},
    "max_stocks": {},
    "fridge_cols": None,
    "fridge_rows": None
} 

#_________________________________Functions____________________________________
def get_stock_tag_names():
    """
    Create stock tag names.
    
    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    # Prepare
    global eans2labels
    global stock_tag_names
    
    # Obtain tags
    for ean in eans2labels:
        new_tag_names = {}
        new_tag_names["ean"] = ean
        new_tag_names["name"] = eans2labels[ean]
        new_tag_names["current_s_tag_name"] = ean + "_current_stock "
        new_tag_names["min_s_tag_name"] = ean + "_min_stock "
        new_tag_names["max_s_tag_name"] = ean + "_max_stock "
        stock_tag_names.append(new_tag_names)

@app.route('/', methods =["GET", "POST"])
def connect_to_network():
    """
    Connect to network.

    Parameters
    ----------
    None.
    
    Returns
    -------
    Rendered Template
        Form 1 rendered.

    """
    # Prepare
    global internet_check_thread
    internet_check_thread = InternetCheckThread.InternetCheckThread()
    
    # Connect to network
    if request.method == "POST":       
        network_name = request.form.get("nname")       
        password = request.form.get("password")
        if network_name != "" and password != "":
            if running_on_intel_nuc:
                os.system("nmcli dev wifi connect {network_name} password {password}".format(network_name=network_name, password=password))
            internet_check_thread.start()
            return redirect(url_for('connecting'))                                 
    return render_template("Form1.html")
 
@app.route('/initialization_form', methods =["GET", "POST"])
def initialization_form():
    """
    Create initialization form.

    Returns
    -------
    Rendered Template
        Form 2 rendered.

    """
    # Prepare
    global form_data
    global form_complete
    global stock_tag_names
    
    # Build form
    if request.method == "POST":       
        form_data["store_name"] = request.form.get("store_name")       
        form_data["store_latitude"] = request.form.get("latitude")
        form_data["store_longitude"] = request.form.get("longitude")
        form_data["store_state"] = request.form.get("state")
        form_data["store_municipality"] = request.form.get("municipality")
        form_data["store_zip_code"] = request.form.get("zip_code")
        form_data["store_address"] = request.form.get("address")
        for dictionary in stock_tag_names:
            form_data["current_stock"][dictionary["ean"]] = request.form.get(dictionary["current_s_tag_name"][:-1])
            form_data["min_stocks"][dictionary["ean"]] = request.form.get(dictionary["min_s_tag_name"][:-1])
            form_data["max_stocks"][dictionary["ean"]] = request.form.get(dictionary["max_s_tag_name"][:-1])
        form_data["fridge_cols"] = request.form.get("num_rows")
        form_data["fridge_rows"] = request.form.get("num_cols")
        form_complete = True        
        return redirect(url_for('success_page'))
    get_stock_tag_names()          
    return render_template("Form2.html", product_block = stock_tag_names)

@app.route('/success_page', methods =["GET"])
def success_page():
    """
    Render success html.

    Returns
    -------
    Rendered Template
        Success page rendered.

    """
    return render_template("Success.html")

@app.route('/connecting', methods =["GET"])
def connecting():
    """
    Render connecting page.

    Returns
    -------
    Rendered Template
        Connecting page rendered.

    """        
    return render_template("Connecting.html")

@app.route('/check_internet_thread_status')
def check_internet_thread_status():    
    """
    Check for internet connection.

    Returns
    -------
    Redirect
        Redirect for URL.

    """
    if internet_check_thread.is_running:
        return redirect(url_for('connecting'))
    else:
        if internet_check_thread.is_connected_to_internet:
            return redirect(url_for('initialization_form'))
        else:
            return redirect(url_for('connect_to_network'))

def load_form(soda_eans2labels, running_on_nuc):
    """
    Load form.

    Parameters
    ----------
    soda_eans2labels : dict
        Dictionary from EANS to labels.
        
    running_on_nuc : bool
        True if running on NUC.

    Returns
    -------
    server : TYPE
        DESCRIPTION.

    """
    # Prepare
    global app
    global server
    global eans2labels
    global running_on_intel_nuc
    eans2labels = soda_eans2labels
    running_on_intel_nuc = running_on_nuc
    server.start()
    time.sleep(1)
    
    # Connect
    webbrowser.open("http://127.0.0.1:7000/")          
    return server

#____________________________________Main______________________________________
if __name__=='__main__':             
    server = ServerThread(app)
    server.start()
    webbrowser.open("http://127.0.0.1:7000/")  
    print("st after server start")
    time.sleep(3)
    server.shutdown()