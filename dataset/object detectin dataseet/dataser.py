# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="9tNjWRx0j6bQrQkINaWt")
project = rf.workspace("dev-virani-nmbd1").project("dp-jughu")
version = project.version(1)
dataset = version.download("yolov9")