import argparse
from lib.detection_grpc import (MLServer, Servicer)
from lib.detection import (DetectionPredictor,
                           chainercv_preprocess,
                           chainercv_postprocess_change_labels,
                           )
import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300, SSD512


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="localhost",
                        help='sergver address')
    parser.add_argument('--port', '-p', default=50080, type=int,
                        help='port')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='gpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg()
    # Load detection model
    model = SSD512(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='voc0712')

    if args.gpu >=0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup server
    detection_server = MLServer(
        Servicer(
            DetectionPredictor(
                model,
                chainercv_preprocess,
                chainercv_postprocess_change_labels
                )
            )
        )

    # Run the server
    detection_server.start(8888)
