import fiftyone as fo
import fiftyone.zoo as foz

# List available zoo datasets
print(foz.list_zoo_datasets())
classes = ["bicycle", "boat", "bus", "car",
           "cat", "dog", "motorcycle", "person"]
#
# Load the COCO-2017 validation split into a FiftyOne dataset
#
# This will download the dataset from the web, if necessary
#
ds = tfds.load('coco', split='train', shuffle_files=True)


# Give the dataset a new name, and make it persistent so that you can
# work with it in future sessions
dataset.name = "coco-2017-validation-example"
dataset.persistent = True

# Visualize the in the App
session = fo.launch_app(dataset)