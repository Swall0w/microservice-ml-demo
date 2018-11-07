# microservice-ml-demo
Micro service architecture demo using rest API

Currently, we use object detection algorithm for demo, but you can change it as you like.

You can find this project's slide -> [Plone Conference 2018: Micro Service Architecture with Machine Learning Application
](https://speakerdeck.com/swall0w/micro-service-architecture-with-machine-learning-application)

## What is inside in this repository?
- REST API server, which is connected to Object Detection server inside it.
- Object Detection server with grpc connection
- A object detection test script

## Requirements
- Python 3.6.6
- ChainerCV 0.8.0
- grpc

## How to use
1. Launch both servers named 'run_api_server.py' and 'run_detection_server.py'
2. You can access to the api server bellow command.
```
$ curl -X POST -F image=@image/dog.jpg 'http://localhost:5000/predict'
# You can get these result.
{"predictions":[{"bbox":{"xmax":"402","xmin":"193","ymax":"388","ymin":"108"},"class":"dog","probability":"0.9997122883796692"}],"success":true}
```

## LICENSE
[MIT](LICENSE)
