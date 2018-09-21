from lib import detection_grpc
from lib.detection import chainercv_parse


if __name__ == '__main__':
    client = detection_grpc.MLClient('localhost:8888',
                                     parse=chainercv_parse)

    with open('dog.jpg', 'rb') as f:
        image = f.read()

    ret = client.predict(image)
    print(ret)
