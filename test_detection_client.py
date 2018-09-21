from lib import detection_grpc


if __name__ == '__main__':
    client = detection_grpc.MLClient('localhost:8888')

    with open('dog.jpg', 'rb') as f:
        image = f.read()

    ret = client.predict(image)
    print(ret)
