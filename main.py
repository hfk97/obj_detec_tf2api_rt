import threading
import itertools
import time
import sys

#this will later be used to add a loading ... to some text
done=False
def dotdotdot(text):
    for c in itertools.cycle(['.', '..', '...','']):
        if done:
            break
        sys.stdout.write('\r'+text+c)
        sys.stdout.flush()
        time.sleep(0.3)
    sys.stdout.write('\nDone!')


# prepare a loading message
t = threading.Thread(target=dotdotdot, args=("Loading required modules",))

# starting loading... thread
t.start()


#needed for the next function
import subprocess
import importlib

# function that imports a library if it is installed, else installs it and then imports it
def getpack(package):
    try:
        return (importlib.import_module(package))
        # import package
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", package],
  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (importlib.import_module(package))
        # import package

def getpack(installname,package):
    try:
        return (importlib.import_module(package))
        # import package
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", installname],
  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (importlib.import_module(package))
        # import package



import numpy as np
import os
import sys
import tensorflow as tf
import pathlib

pwd=subprocess.check_output("pwd",shell=True).rstrip()
os.chdir("./models/research")

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.chdir(pwd)

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

done=True
time.sleep(0.3)
print("\n")

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model



# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def run_inference(model):
    # activate video capture option
    cv2 = getpack("opencv-python", "cv2")
    cap = cv2.VideoCapture(0)

    while True:
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        #print(output_dict)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('object detection', cv2.resize(image_np,(800,600)))
        if cv2.waitKey(25) & 0xFF ==ord('q'):
            cv2.destroyAllWindows()
            break


def main():
    while True:
        choice=int(input("Do you want to run a model without instance segmentation (0) or with instance segementation (1)? If you want to choose another model press (3).\nBe aware that instance segmentation is computationally intensive.\nTo exit the program choose (0): "))
        if choice == 1:
            print("To end this live object detection press (q).")
            #Without instance segmentation
            model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
            detection_model = load_model(model_name)
            run_inference(detection_model)

        elif choice == 2:
            print("To end this live object detection press (q).")
            #With instance segmentation
            model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
            masking_model = load_model("mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28")
            masking_model.output_shapes
            run_inference(masking_model)

        elif choice == 3:
            model_name=input("Please copy paste the name of your selected COCO-trained model (see here https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): ")
            detection_model = load_model(model_name)
            run_inference(detection_model)

        else:
            print("Thank you.\nGoodbye.")
            break

if __name__=="__main__":
    main()
