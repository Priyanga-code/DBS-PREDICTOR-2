from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("dbs.jl")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        sgd_value = float(request.form.get("q"))
        prediction = model.predict(np.array([[sgd_value]]))[0][0]
        return render_template("prediction.html", r=round(prediction, 2))
    except:
        return render_template("prediction.html", r="Invalid input")

if __name__ == "__main__":
    app.run(debug=True)
