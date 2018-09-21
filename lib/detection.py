from skimage import io as skio
import numpy as np
from PIL import Image
from chainercv.datasets import voc_bbox_label_names


def chainercv_parse(data):
    predictions = []
    for index in range(len(data.boxes)):
        r = {"class": data.classes[index],
             "bbox": {
                 "ymin": str(int(data.boxes[index].box[0])),
                 "xmin": str(int(data.boxes[index].box[1])),
                 "ymax": str(int(data.boxes[index].box[2])),
                 "xmax": str(int(data.boxes[index].box[3]))
             },
             "probability": str(float(data.confs[index]))
             }
        predictions.append(r)

    return predictions


class DetectionPredictor(object):
    def __init__(self, model, preprocess=None, postprocess=None):
        self.__model = model
        self.__preprocess = preprocess
        self.__postprocess = postprocess

    def predict(self, image):
        """Detection objects on bytes images

        Args:
            image: bytes image
        """

        if self.__preprocess != None:
            image = self.__preprocess(image)

        result = self.__model.predict(image)

        if self.__postprocess != None:
            result = self.__postprocess(result)

        return result


def chainercv_preprocess(image):
    """Convert bytes image to numpy array"""
    image = skio.imread(image)
    image = image.transpose(2, 0, 1)
    return [image]


def chainercv_postprocess_pack_each_item(results):
    """Convert chainerCV prediction to the defined json format"""
    bboxes, labels, scores = results

    # loop over the results and add them to the list of
    # returned predictions
    predictions = []
    for index, bbox in enumerate(bboxes[0]):
        r = {"class": str(voc_bbox_label_names[int(labels[0][index])]),
             "bbox": {
                 "ymin": str(bbox[0]),
                 "xmin": str(bbox[1]),
                 "ymax": str(bbox[2]),
                 "xmax": str(bbox[3])
             },
             "probability": str(scores[0][index])
             }
        predictions.append(r)

    return predictions


def chainercv_postprocess_change_labels(results):
    """Convert chainerCV prediction to the defined json format"""
    bboxes, labels, scores = results
    # loop over the results and add them to the list of
    # returned predictions
    classes = []
    boxes = []
    confs = []
    for index, bbox in enumerate(bboxes[0]):
        classes.append(str(voc_bbox_label_names[int(labels[0][index])]))
        boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        confs.append(scores[0][index])

    return (boxes, classes, confs)
