from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/analysis', methods=['post'])
def imgAnalysis():
    srcFileName = request.form.get('source_image')
    path = "F:/demo/res/img/"
    srcFilePath = path + srcFileName


if __name__ == '__main__':
    app.run(debug=False, port = 8442)
