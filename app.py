from threading import Thread
import pickle 
from flask import Flask, jsonify, request,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from ubuntuclf.model import run, load_model

tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3))
app=Flask(__name__,template_folder='templates')
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/train")
def train_new_model():
    trainer_job = Thread(target=run)
    trainer_job.start()
    return "training started !!"    

@app.route("/load_model")
def load_new_model():
    cur_model = load_model()
    return "model updated !"


@app.route("/predict", methods = ["post"])
def predict():
    data = request.json['data']
    prediction = f'This text belongs to {model.predict[0]} with probability {max(model.predict_proba[0])}'
    return jsonify({
        "prediction"
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True)