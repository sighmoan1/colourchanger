import os
import numpy as np
from PIL import Image as PIL_Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def colour_distance(colour1, colour2):
    """Returns the Euclidean distance between two RGB colours"""
    r1, g1, b1 = colour1
    r2, g2, b2 = colour2
    return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def is_similar_colour(colour1, colour2, threshold=50):
    """Returns True if the Euclidean distance between two RGB colours is within the given threshold"""
    return colour_distance(colour1, colour2) <= threshold


def process_image(args):
    """Processes a single image"""
    input_path, output_path, background_colour, threshold_distance, new_colour, file_extension = args
    try:
        # Open the input image and convert it to a numpy array
        input_image = PIL_Image.open(input_path).convert("RGBA")
        input_array = np.array(input_image)

        # Create a new RGBA array with a white background
        output_array = np.full_like(input_array, [255, 255, 255, 255], dtype=np.uint8)

        # Create a boolean mask comparing the input_array RGB values with the background_colour
        mask = np.apply_along_axis(lambda c: is_similar_colour(c[:3], background_colour, threshold_distance), axis=-1, arr=input_array)

        # Use the mask to combine input_array and output_array
        output_array = np.where(mask[..., None], np.array(new_colour + [255], dtype=np.uint8), input_array)

        # Convert the output array back to an image and save it to the output folder
        output_image = PIL_Image.fromarray(output_array, mode="RGBA")
        output_image.save(output_path)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


if __name__ == '__main__':
    print("Hello! This script processes a set of images and replaces a specified background color with a new color.")
    print("The script will ask you for the following input:")
    print("  a. Input folder containing images")
    print("  b. Output folder for processed images")
    print("  c. Background color to replace in RGB format (e.g., 0 0 0)")
    print("  d. Threshold distance for color similarity (default: 50)")
    print("  e. New color to replace the background color in RGB format (e.g., 255 255 255)")
    print("  f. File extension to process (default: png)")

    user_choice = input("Do you want to continue with the script? Type 'q' to quit or press enter to continue: ")

    if user_choice.lower() != 'q':
        input_folder = input("Enter the input folder containing images: ")
        output_folder = input("Enter the output folder for processed images: ")
        background_colour_input = input("Enter the background colour to replace in RGB format (default: 0 0 0): ")
        background_colour = list(map(int, background_colour_input.split())) if background_colour_input else [0, 0, 0]
        threshold_distance_input = input("Enter the threshold distance for colour similarity (default: 50): ")
        threshold_distance = int(threshold_distance_input) if threshold_distance_input else 50
        new_colour_input = input("Enter the new colour to replace the background colour in RGB format (default: 255 255 255): ")
        new_colour = list(map(int, new_colour_input.split())) if new_colour_input else [255, 255, 255]
        file_extension = input("Enter the file extension to process (default: png): ") or "png"

        # Check if the output folder exists and create it if it doesn't
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create a pool of worker processes
        num_processes = min(cpu_count(), len(os.listdir(input_folder)))
        with Pool(num_processes) as pool:
            # Loop through all files with the given extension in the input folder
            filenames = [filename for filename in os.listdir(input_folder) if filename.endswith("." + file_extension)]   
            with tqdm(total=len(filenames), desc="Processing images", unit="image") as pbar:
                for filename in filenames:
                    # Build paths to input and output images
                    input_path = os.path.join(input_folder, filename)
                    output_filename = filename
                    output_path = os.path.join(output_folder, output_filename)

                    # Process the image
                    process_image((input_path, output_path, background_colour, threshold_distance, new_colour, file_extension))
                    pbar.update(1)

    elif user_choice.lower() == 'q':
        print("Exiting the script. Have a great day!")
    else:
        print("Invalid input. Please run the script again and enter 'continue' or 'exit'.")




