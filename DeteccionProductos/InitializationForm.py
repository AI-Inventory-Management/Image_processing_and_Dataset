from flask import Flask, jsonify, request, render_template, redirect, url_for, Markup
import ServerThread
import InternetCheckThread
import webbrowser
import time
import os

app = Flask(__name__)
server = ServerThread.ServerThread(app)
internet_check_thread = None
form_complete = False
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

def get_stock_tag_names():
    global eans2labels
    global stock_tag_names
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
    global internet_check_thread
    internet_check_thread = InternetCheckThread.InternetCheckThread()
    if request.method == "POST":       
        network_name = request.form.get("nname")       
        password = request.form.get("password")
        if network_name != "" and password != "":
            #os.system("nmcli dev wifi connect {network_name} password {password}".format(network_name=network_name, password=password))
            internet_check_thread.start()
            return redirect(url_for('connecting'))                                 
    return render_template("Form1.html")
 
@app.route('/initialization_form', methods =["GET", "POST"])
def initialization_form():
    global form_data
    global form_complete
    global stock_tag_names
    
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
    return "success :)"

@app.route('/connecting', methods =["GET"])
def connecting():        
    return render_template("Connecting.html")

@app.route('/check_internet_thread_status')
def check_internet_thread_status():    
    if internet_check_thread.is_running:
        return redirect(url_for('connecting'))
    else:
        if internet_check_thread.is_connected_to_internet:
            return redirect(url_for('initialization_form'))
        else:
            return redirect(url_for('connect_to_network'))

def load_form(soda_eans2labels):
    global app
    global server
    global eans2labels
    eans2labels = soda_eans2labels
    server.start()
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:7000/")          
    return server


if __name__=='__main__':       
    #initialization_form_server = multiprocessing.Process(target= app.run(host = '0.0.0.0', port = 7000, debug = True) )                 
    #app.run(host = '0.0.0.0', port = 7000, debug = True)        
    server = ServerThread(app)
    server.start()
    webbrowser.open("http://127.0.0.1:7000/")  
    print("st after server start")
    time.sleep(3)
    server.shutdown()


   
