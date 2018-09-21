import os
from concurrent import futures

import grpc
import time

from lib import detection_pb2
from lib import detection_pb2_grpc
import io

CHUNK_SIZE = 1024 * 1024  # 1MB


def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if len(piece) == 0:
                return
            yield detection_pb2.Chunk(buffer=piece)


def get_virtual_file_chunks(virtualfile):
    virtualfile = io.BytesIO(virtualfile)
    while True:
        piece = virtualfile.read(CHUNK_SIZE);
        if len(piece) == 0:
            return
        yield detection_pb2.Chunk(buffer=piece)


def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)


def save_chunks_to_file_object(chunks):
    output = io.BytesIO()
    for chunk in chunks:
        output.write(chunk.buffer)
    return output


class MLClient(object):
    def __init__(self, address, parse=None):
        channel = grpc.insecure_channel(address)
        self.stub = detection_pb2_grpc.MLServerStub(channel)
        self.__parse = parse

    def predict(self, in_file):
        chunks_generator = get_virtual_file_chunks(in_file)
        response = self.stub.predict(chunks_generator)
        if self.__parse != None:
            response = self.__parse(response)
        return response


class Servicer(detection_pb2_grpc.MLServerServicer):
    def __init__(self, predictor):
        self.__predictor = predictor

    def predict(self, request_iterator, context):
        image = save_chunks_to_file_object(request_iterator)
        # predict
        boxes, classes, confs  = self.__predictor.predict(image)
        bboxes = [detection_pb2.Boundingbox(box=box) for box in boxes]
        return detection_pb2.Reply(
            boxes=bboxes, classes=classes, confs=confs)


class MLServer(detection_pb2_grpc.MLServerServicer):
    def __init__(self, servicer):
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1))
        detection_pb2_grpc.add_MLServerServicer_to_server(
            servicer, self.server)

    def start(self, port):
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()

        try:
            while True:
                time.sleep(60*60*24)
        except KeyboardInterrupt:
            self.server.stop(0)
