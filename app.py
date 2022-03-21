from flask import Flask, request, jsonify

from service.AnalysisService import analysis

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/analysis', methods=['post'])
def imgAnalysis():
    srcFileName = request.form.get('source_image')
    path = "F:/demo/res/img/"
    srcFilePath = path + srcFileName
    fileName, yang, ying = analysis(srcFilePath)
    tps = round(yang / (yang + ying) * 100, 1)
    return jsonify([fileName, yang, ying, tps])


if __name__ == '__main__':
    app.run()
