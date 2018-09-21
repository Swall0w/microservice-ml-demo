import flask
import argparse
import io
from lib import detection_grpc
from lib.detection import chainercv_parse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--host', default='127.0.0.1', type=str)
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()

app = flask.Flask(__name__)


object_detection = detection_grpc.MLClient('localhost:8888',
                                           parse=chainercv_parse)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()

            # send to server from client
            predictions = object_detection.predict(image)

            # return result
            data["predictions"] = predictions
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print("Run server")
    app.run(host=args.host, port=args.port)
