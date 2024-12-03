import os
import easyocr
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time  
import pandas as pd

# Function to display the image
def display_image(img, img_name=None, figsize=(10, 10)):
    """
    Displays an image with optional title and customizable size.
    """
    plt.figure(figsize=figsize)  # Set the figure size
    plt.imshow(img)
    plt.axis('off')  # Turn off axis for better visualization
    if img_name:
        plt.title(img_name)  # Optional: Add a title
    plt.show()

# OCR Processing Function with Confidence Threshold and Bounding Box Coordinates
def process_image(image_path, output_dir, reader, confidence_threshold=0.8):
    """
    Processes a single image using EasyOCR with a confidence threshold.
    Annotates the image with bounding boxes and saves OCR results to a CSV file.
    """
    # Start timing
    start_time = time.time()

    # Load the image
    image = Image.open(image_path)

    # Perform OCR
    results = reader.readtext(image_path)

    # Annotate the image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    text_output = []  # Initialize a list to hold OCR result data for CSV

    for (bbox, text, confidence) in results:
        # Filter results based on confidence threshold
        if confidence >= confidence_threshold:
            # Draw bounding box
            top_left = tuple(bbox[0])
            bottom_right = tuple(bbox[2])
            draw.rectangle([top_left, bottom_right], outline="blue", width=3)
            
            # Annotate text
            draw.text((top_left[0], bottom_right[1] + 5), text, fill="red")  # Place text below the bounding box
            
            # Append text, confidence, and bounding box to the output
            text_output.append([text, bbox, confidence])

    # Generate output file names
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    annotated_image_path = os.path.join(output_dir, f"Annotated_{base_name}.png")
    csv_output_path = os.path.join(output_dir, f"OCR_{base_name}.csv")

    # Save annotated image
    annotated_image.save(annotated_image_path)

    # Save text output to CSV
    if text_output:
        df = pd.DataFrame(text_output, columns=["Text", "Bounding Box", "Confidence"])
        df.to_csv(csv_output_path, index=False, encoding='utf-8')

    # End timing and calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Processed: {image_path} in {elapsed_time:.2f} seconds")

# Main script to process all PNG images
def process_all_images(input_dir, output_dir, confidence_threshold=0.8, reader_language="en"):
    """
    Processes all PNG images in the input directory with a confidence threshold.
    Saves annotated images and OCR results to the output directory in CSV format.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize EasyOCR reader
    reader = easyocr.Reader([reader_language])

    # Iterate over all PNG files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.png'):
            image_path = os.path.join(input_dir, file_name)
            process_image(image_path, output_dir, reader, confidence_threshold)

# Specify input and output directories
input_dir = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Images'

# Process all images with a confidence threshold of 0.0 for English
output_dir_en_0 = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Processed_EN_0_Conf'
process_all_images(input_dir, output_dir_en_0, confidence_threshold=0.0, reader_language="en")

# Process all images with a confidence threshold of 0.80 for English
output_dir_en_80 = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Processed_EN_80_Conf'
process_all_images(input_dir, output_dir_en_80, confidence_threshold=0.80, reader_language="en")

# Process all images with a confidence threshold of 0.0 for Turkish
output_dir_tr_0 = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Processed_TR_0_Conf'
process_all_images(input_dir, output_dir_tr_0, confidence_threshold=0.0, reader_language="tr")

# Process all images with a confidence threshold of 0.80 for Turkish
output_dir_tr_80 = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Processed_TR_80_Conf'
process_all_images(input_dir, output_dir_tr_80, confidence_threshold=0.80, reader_language="tr")
