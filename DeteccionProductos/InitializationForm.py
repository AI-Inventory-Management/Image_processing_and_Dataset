from flask import Flask, request, render_template, redirect, url_for, Markup
import ServerThread
import webbrowser
import time
import os

app = Flask(__name__)
server = ServerThread.ServerThread(app)
form_complete = False
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

'''                    
for soda in self.soda_labels:
    print('')
    print('-------------- {s} --------------'.format(s=soda))
    current_stock[self.ean[i]] = int(input("how many {s} are on the store right now: ".format(s=soda)))
    min_stocks[self.ean[i]] = int(input("whats the min of {s} to generate an alert: ".format(s=soda)))
    max_stocks[self.ean[i]] = int(input("whats amount of {s} to fill the store: ".format(s=soda)))
                        
    i += 1
'''

@app.route('/', methods =["GET", "POST"])
def connect_to_network():
    #global server    
    if request.method == "POST":       
        network_name = request.form.get("nname")       
        password = request.form.get("password")
        #os.system("nmcli dev wifi connect {network_name} password {password}".format(network_name=network_name, password=password))
        if network_name != "" and password != "":
            #server.shutdown()
            return redirect(url_for('initialization_form'))                                 
    return render_template("Form1.html")
 
@app.route('/initialization_form', methods =["GET", "POST"])
def initialization_form():
    global form_data
    global form_complete

    product_html_code = '''
    <p class="product-name">{product_name}</p>
        <table>
          <tbody>
            <tr>
              <td class="label-text">Stock Actual</td>
              <td class="label-text">Stock Mínimo</td>
              <td class="label-text">Stock Máximo</td>
            </tr>
            <tr>              
              <td><input type="number" name={product_ean_current_stock}/></td>
              <td><input type="number" name={product_ean_min_stock}/></td>
              <td><input type="number" name={product_ean_max_stock}/></td>
            </tr>
          </tbody>
        </table>
    '''.format(product_name = "totis de andrea", product_ean_current_stock="\"6848564646\"_current", product_ean_min_stock="\"6848564646\"_min", product_ean_max_stock="\"6848564646\"_max")
    if request.method == "POST":       
        form_data["store_name"] = request.form.get("nname")       
        form_data["store_latitude"] = request.form.get("Latitud")
        form_data["store_longitude"] = request.form.get("Longitud")
        form_data["store_state"] = request.form.get("Estado")
        form_data["store_municipality"] = request.form.get("Municipio")
        form_data["store_zip_code"] = request.form.get("CP")
        form_data["store_address"] = request.form.get("Direccion")
        form_data["current_stock"] = request.form.get("nname")
        form_data["store_municipality"] = request.form.get("nname")
        form_data["store_municipality"] = request.form.get("nname")
        form_data["fridge_cols"] = request.form.get("Columnas")
        form_data["fridge_rows"] = request.form.get("Filas")
        form_complete = True        
        return redirect(url_for('success_page'))          
    return render_template("Form2.html", product_block = Markup(product_html_code) )

@app.route('/succes_page', methods =["GET"])
def succes_page():    
    return "succes :)"

def load_form():
    global app
    global server
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


   
