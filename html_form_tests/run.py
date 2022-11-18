
# importing Flask and other modules
from flask import Flask, request, render_template, redirect, url_for
import webbrowser
import time
import os

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def connect_to_network():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       network_name = request.form.get("nname")
       # getting input with name = lname in HTML form
       password = request.form.get("password")
       #os.system("nmcli dev wifi connect {network_name} password {password}".format(network_name=network_name, password=password))
       return redirect(url_for('initialization_form'))
    return render_template("Form1.html")
 
@app.route('/initialization_form', methods =["GET", "POST"])
def initialization_form():
    return render_template("Form2.html")

if __name__=='__main__':
   time.sleep(5)   
   webbrowser.open("http://127.0.0.1:7000/")
   app.run(host = '0.0.0.0', port = 7000, debug = True)   
   