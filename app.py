import flask

from aircraft import transform_image
from predict import predict_label

app = flask.Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.method == "POST":
        if flask.request.get_json().get("image"):
            image = transform_image(flask.request.get_json().get("image"))
            label = predict_label(image)

            response["label"] = label
            response["success"] = True

    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()
