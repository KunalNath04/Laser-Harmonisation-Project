import pixellib
from pixellib.instance import instance_segmentation
import pixellib
from pixellib.custom_train import instance_custom_training

def train_pixellib_model(dataset_path, num_classes, model_name):
    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= num_classes)
    train_maskrcnn.load_dataset(dataset_path)
    train_maskrcnn.train_model(num_epochs = 300, augmentation=True, path_trained_models = model_name)

def detect_objects_in_video(video_path, model_path, output_video_name):
    segment_video = instance_segmentation()
    segment_video.load_model(model_path)
    segment_video.process_video(video_path, frames_per_second= 15, output_video_name=output_video_name)

# Train the Pixellib model
train_pixellib_model(r'C:\Users\kunal\OneDrive\Desktop\SEM 4\dp\dataset\plus\plus\images', 2, "my_trained_model")

# Detect objects in a video
detect_objects_in_video( r"Downloads\WhatsApp Video 2024-04-24 at 11.10.56 PM.mp4", "my_trained_model/mask_rcnn_model.h5", "output_video.mp4")