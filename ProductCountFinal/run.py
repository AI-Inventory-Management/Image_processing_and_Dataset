from flask import Flask, render_template
import requests

app = Flask(__name__)

app.secret_key = "Confidential_secret"

recieved_data = {}

@app.route('/', methods = ['GET'])
def home():
    send_data()
    return render_template('home_template.html')

@app.route('/send_data', methods=['GET', 'POST'])
def send_data():
    data = {"main_data":"some data on json"}
    res = requests.post('http://127.0.0.1:7000/constant_messages', json=data)
    if res.ok:
        print("printing data from HW:")
        print(res.json())

if __name__ == '__main__':
	app.run(host = '0.0.0.0',port = 5000, debug = True)