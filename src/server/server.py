from flask import Flask, redirect, url_for, request, jsonify, abort
import src.Predictor.Predictor

predictor = None

app = Flask(__name__)


@app.before_first_request
def initiate_predictor():
    global predictor
    predictor = src.Predictor.Predictor.Predictor(ad_reprs_address="../../representations/ad_reprs.pt",
                                                  id_to_package_address="../../representations/id_to_package.pkl",
                                                  query_ad_coordinator_checkpoint="../../representations"
                                                                                  "/QueryAdCoordinator_checkpoint.pt")


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/predict', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        query = request.form['query']
    else:
        query = request.args.get('query')
    if type(query) is not str:
        abort(400)
    packages = predictor.predict_from_query(query)
    return jsonify({"related_packages": packages})


if __name__ == '__main__':
    app.run(debug=True)
