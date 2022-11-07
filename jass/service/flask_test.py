
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<h2>Hello, World!</h2>'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)