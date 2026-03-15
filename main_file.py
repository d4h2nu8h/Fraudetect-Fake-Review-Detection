from flask import Flask
from flask import Flask, app, request
import json
from base64 import b64decode, b64encode
from flask_cors import CORS, cross_origin
import requests
app = Flask(__name__)
cors = CORS(app)
ngrok = "https://7304-34-125-23-163.ngrok-free.app"
@app.route('/fake_review', methods=['GET','POST'])
def a():
    uri_data = request.json
    print(uri_data)
    x = [str(i) for i in uri_data.values()]
    img = ','.join(x)
    print(img, type(img))
    r = requests.post(f"{ngrok}/fake_review", data={'uri':img})
    
    return r.json()
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)