import os
import random
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# Set the size of the Cartesian plane and number of plus signs to generate
plane_size = 2000
num_signs = 1000
arm_length = 200  # Adjust as needed
line_width = 1.5  # Set the desired line width for the plus sign

# Create a directory to store the generated images and annotations
output_dir = 'images_with_annotations'
os.makedirs(output_dir, exist_ok=True)

# Generate and save each plus sign with bounding box and XML annotation
for i in range(num_signs):
    # Generate random center coordinates within the bounds of the plane
    x0 = random.uniform(-(plane_size-300)/2, (plane_size-300)/2)
    y0 = random.uniform(-(plane_size-300)/2, (plane_size-300)/2)

    # Compute bounding box coordinates
    xmin = int(x0 - arm_length)
    xmax = int(x0 + arm_length)
    ymin = int(y0 - arm_length)
    ymax = int(y0 + arm_length)

    # Create XML annotation for the image
    annotation = ET.Element('annotation')

    # Add filename to annotation
    filename = f'plus_sign_{i}.png'
    ET.SubElement(annotation, 'filename').text = filename

    # Add size (width and height) to annotation
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(plane_size)
    ET.SubElement(size, 'height').text = str(plane_size)
    ET.SubElement(size, 'depth').text = '3'  # Assuming RGB images

    # Add object (plus sign) to annotation
    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = 'plus sign'
    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(xmin)
    ET.SubElement(bbox, 'ymin').text = str(ymin)
    ET.SubElement(bbox, 'xmax').text = str(xmax)
    ET.SubElement(bbox, 'ymax').text = str(ymax)

    # Create ElementTree object and write to XML file
    xml_tree = ET.ElementTree(annotation)
    xml_path = os.path.join(output_dir, f'plus_sign_{i}.xml')
    xml_tree.write(xml_path)

    # Plotting the plus sign with bounding box
    plt.figure(figsize=(1, 1))
    plt.plot([xmin, xmax], [y0, y0], 'b', linewidth=line_width)  # Horizontal line
    plt.plot([x0, x0], [ymin, ymax], 'b', linewidth=line_width)  # Vertical line
    plt.axis('off')
    plt.xlim(-plane_size/2, plane_size/2)
    plt.ylim(-plane_size/2, plane_size/2)
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the image
    image_path = os.path.join(output_dir, filename)
    plt.savefig(image_path)
    plt.close()  # Close the plot to free memory

print(f"Generated {num_signs} plus signs with bounding boxes and saved images and annotations to {output_dir}.")
