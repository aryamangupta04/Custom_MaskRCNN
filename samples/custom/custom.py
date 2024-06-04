import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import skimage.io
from mrcnn.utils import Dataset
from mrcnn.config import Config
import mrcnn.model as modellib

class CustomConfig(Config):
    NAME = "custom"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 +   # Background + number of classes (here, 2)

    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

class CustomDataset(Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only two classes to add.
        self.add_class("custom", 1, "warning traffic light")
        self.add_class("custom", 2, "stop traffic light")
        self.add_class("custom",3,"go traffic light")

        # Path to annotations and images
        annotations_dir = os.path.join(dataset_dir, 'annotations', subset)
        images_dir = os.path.join(dataset_dir, 'images')

        # Iterate over all annotation files
        for annotation_file in os.listdir(annotations_dir):
            if annotation_file.endswith('.json'):
                self.load_json_annotation(images_dir, annotations_dir, annotation_file)
            elif annotation_file.endswith('.xml'):
                self.load_xml_annotation(images_dir, annotations_dir, annotation_file)

    def load_json_annotation(self, images_dir, annotations_dir, annotation_file):
        json_path = os.path.join(annotations_dir, annotation_file)
        with open(json_path) as f:
            annotation = json.load(f)

        # Assuming 'description' field contains the image filename
        image_filename = annotation['description']
        image_path = os.path.join(images_dir, image_filename)
        image = skimage.io.imread(image_path)
        height, width = annotation['size']['height'], annotation['size']['width']

        polygons = []
        labels = []

        for obj in annotation['objects']:
            if obj['geometryType'] == 'rectangle':
                x1, y1 = obj['points']['exterior'][0]
                x2, y2 = obj['points']['exterior'][1]
                polygons.append([x1, y1, x2, y2])
                if obj['classTitle'] == 'warning traffic light':
                    labels.append(1)
                elif obj['classTitle'] == 'stop traffic light':
                    labels.append(2)

        self.add_image(
            "custom",
            image_id=image_filename,  # Use the image filename as the image ID
            path=image_path,
            width=width,
            height=height,
            polygons=polygons,
            labels=labels
        )

    def load_xml_annotation(self, images_dir, annotations_dir, annotation_file):
        xml_path = os.path.join(annotations_dir, annotation_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_filename = root.find('filename').text
        image_path = os.path.join(images_dir, image_filename)
        image = skimage.io.imread(image_path)
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        polygons = []
        labels = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name == 'warning traffic light':
                label = 1
            elif class_name == 'stop traffic light':
                label = 2
            elif class_name == 'go traffic light':
                label=3
            else:
                continue

            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            polygons.append([x1, y1, x2, y2])
            labels.append(label)

        self.add_image(
            "custom",
            image_id=image_filename,  # Use the image filename as the image ID
            path=image_path,
            width=width,
            height=height,
            polygons=polygons,
            labels=labels
        )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        polygons = info['polygons']
        labels = info['labels']
        
        # Convert polygons to a bitmap mask of shape
        count = len(polygons)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        
        for i, (x1, y1, x2, y2) in enumerate(polygons):
            mask[y1:y2, x1:x2, i] = 1

        # Return mask and array of class IDs of each instance
        return mask, np.array(labels, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

# Example usage
dataset = CustomDataset()
dataset.load_custom(dataset_dir='path_to_dataset', subset='train')
dataset.prepare()

# Print a few data points to verify
for image_id in dataset.image_ids[:5]:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    print(f"Image ID: {image_id}, Class IDs: {class_ids}")

class CustomModel:
    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir

    def train(self, dataset_train, dataset_val):
        model = modellib.MaskRCNN(mode="training", config=self.config,
                                  model_dir=self.model_dir)
        model.train(dataset_train, dataset_val,
                    learning_rate=self.config.LEARNING_RATE,
                    epochs=30,
                    layers='heads')

# Parse command line arguments
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = CustomModel(config, args.logs)
        dataset_train = CustomDataset()
        dataset_train.load_custom(args.dataset, "train")
        dataset_train.prepare()

        dataset_val = CustomDataset()
        dataset_val.load_custom(args.dataset, "val")
        dataset_val.prepare()

        model.train(dataset_train, dataset_val)
    else:
        model = CustomModel(config, args.logs)
        model.load_weights(args.weights)
