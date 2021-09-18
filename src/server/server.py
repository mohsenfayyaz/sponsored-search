from flask import Flask, redirect, url_for, request, jsonify, abort, render_template
import requests
from bs4 import BeautifulSoup
import src.Predictor.Predictor

predictor = None

app = Flask(__name__)


@app.before_first_request
def initiate_predictor():
    global predictor
    predictor = src.Predictor.Predictor.Predictor(ad_reprs_address="../../representations/ad_reprs.pt",
                                                  vocab_reprs_address="../../representations/vocab_reprs.pt",
                                                  id_to_package_address="../../representations/ad_id_to_package.pkl")


@app.route('/')
def hello():
    return redirect('/predict?query="زامبی"')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        query = request.form['query']
    else:
        query = request.args.get('query')
    if type(query) is not str:
        abort(400)
    packages = predictor.predict_from_query(query)
    return jsonify({"related_packages": packages})


@app.route('/predict/visual', methods=['POST', 'GET'])
def visual_predict():
    if request.method == 'POST':
        query = request.form['query']
    else:
        query = request.args.get('query')
        k = request.args.get('k')
    if k is None or not 0 < int(k) < 10:
        k = 5
    else:
        k = int(k)
    if type(query) is not str:
        abort(400)
    packages = predictor.predict_from_query(query, k=k)
    metas = extract_meta(packages)
    return render_template('card.html', result=metas)


def extract_meta(packages):
    final_metas = []
    for package in packages:
        new_meta = {}
        url = 'https://cafebazaar.ir/app/' + package
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features="lxml")
        metas = soup.find_all('meta')
        if not does_app_exist(metas):
            final_metas.append({"package": package, "title": "not found", "description": "not found", "image_url": "not found"})
            continue
        new_meta["package"] = package
        new_meta["title"] = \
            [meta.attrs['content'] for meta in metas if
             'property' in meta.attrs and meta.attrs['property'] == 'og:title'][0]
        new_meta["description"] = \
            [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description'][0]
        new_meta["image_url"] = \
            [meta.attrs['content'] for meta in metas if
             'property' in meta.attrs and meta.attrs['property'] == 'og:image'][0]
        final_metas.append(new_meta)
    return final_metas


def does_app_exist(metas):
    title = [meta.attrs['content'] for meta in metas if
             'property' in meta.attrs and meta.attrs['property'] == 'og:title']
    return len(title) > 0


if __name__ == '__main__':
    app.run(debug=True)
