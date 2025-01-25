import cv2
import numpy as np
import os
import random

# Path to the plus sign image
plus_sign_path = '/Users/dvirani/dp/dataset/object detectin dataseet/WhatsApp Image 2024-04-24 at 17.12.47.jpeg'

# Path to the folder containing background images
backgrounds_folder = '/Users/dvirani/dp/dataset/object detectin dataseet/data/background'

# Output folder to save the generated images
output_folder = '/Users/dvirani/dp/dataset/object detectin dataseet/data/output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

try:
    # Load the plus sign image
    plus_sign = cv2.imread(plus_sign_path, cv2.IMREAD_UNCHANGED)

    # Extract the alpha channel if exists
    if plus_sign.shape[-1] == 4:
        alpha_channel = plus_sign[:, :, 3]
        plus_sign = plus_sign[:, :, :3]
    else:
        alpha_channel = None

    # Iterate over each background image
    background_files = os.listdir(backgrounds_folder)
    for i, bg_file in enumerate(background_files):
        background_path = os.path.join(backgrounds_folder, bg_file)

        # Load the background image
        background = cv2.imread(background_path)

        if background is None:
            print(f"Error: Failed to load background image {bg_file}")
            continue

        # Resize the plus sign to a random size (between 10% and 40% of background height)
        scale_factor = random.uniform(0.1, 0.4)
        new_height = int(scale_factor * background.shape[0])
        new_width = int((plus_sign.shape[1] / plus_sign.shape[0]) * new_height)
        plus_sign_resized = cv2.resize(plus_sign, (new_width, new_height))

        # Randomly place the plus sign on the background
        y_pos = random.randint(0, background.shape[0] - plus_sign_resized.shape[0])
        x_pos = random.randint(0, background.shape[1] - plus_sign_resized.shape[1])

        if alpha_channel is not None:
            # Blend the images using the alpha channel
            alpha_s = plus_sign_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                background[y_pos:y_pos+plus_sign_resized.shape[0], x_pos:x_pos+plus_sign_resized.shape[1], c] = (
                    alpha_s * plus_sign_resized[:, :, c] + alpha_l * background[y_pos:y_pos+plus_sign_resized.shape[0], x_pos:x_pos+plus_sign_resized.shape[1], c]
                )
        else:
            # Directly overlay the plus sign without alpha blending
            background[y_pos:y_pos+plus_sign_resized.shape[0], x_pos:x_pos+plus_sign_resized.shape[1]] = plus_sign_resized

        # Save the resulting image
        output_path = os.path.join(output_folder, f'image_{i}.png')
        cv2.imwrite(output_path, background)

        print(f'Processed image {i+1}/{len(background_files)}')

    print('Dataset generation complete.')

except Exception as e:
    print(f"An error occurred: {str(e)}")
