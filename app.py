import json

from flask import Flask, request, jsonify
from multiprocessing import Process

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
    d = {'path': srcFilePath}
    p = Process(target=analysis, kwargs=d)
    p.start()
    p.join()
    with open('./static/record.json', 'r') as recordF:
        record = json.load(recordF)
        fileName = record['fileName']
        yang = record['yang']
        ying = record['ying']
    tps = round(yang / (yang + ying) * 100, 1)
    return jsonify([fileName, yang, ying, tps])


if __name__ == '__main__':
    app.run(port=5000)
