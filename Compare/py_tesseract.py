import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
import os

# Path to tesseract executable (update if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Folder containing the images
image_folder = r"C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Images"
output_base_folder = os.path.join(image_folder, "PyTesseract")
os.makedirs(output_base_folder, exist_ok=True)  # Ensure the base output folder exists

# Confidence thresholds
CONFIDENCE_THRESHOLDS = [0, 80]  # No threshold and 80%

# Languages to process
LANGUAGES = ["tur", "eng"]  # Turkish and English

# Process each PNG file in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(".png"):
        # Full path to the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        for lang in LANGUAGES:
            lang_label = "Turkish" if lang == "tur" else "English"
            
            for threshold in CONFIDENCE_THRESHOLDS:
                threshold_label = f"{threshold}%"

                # Create output folders
                process_folder = os.path.join(output_base_folder, f"{lang_label}_{threshold_label}")
                os.makedirs(process_folder, exist_ok=True)

                # Perform OCR
                data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)

                # Initialize lists for dataframe
                texts, bbox_coords, confidences = [], [], []

                # Filter and store results based on the confidence threshold
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text:  # Skip empty text
                        conf = int(data['conf'][i])
                        if conf >= threshold:
                            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                            # Save the result for the dataframe
                            texts.append(text)
                            bbox_coords.append((x, y, x + w, y + h))
                            confidences.append(conf)

                            # Draw the bounding box and text on the image
                            color = (255, 0, 0) if lang == "tur" else (0, 255, 0)  # Blue for Turkish, Green for English
                            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                            label = text
                            cv2.putText(image, label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Save the annotated image
                annotated_image_path = os.path.join(
                    process_folder, f"Annotated_{os.path.splitext(filename)[0]}.jpg"
                )
                cv2.imwrite(annotated_image_path, image)

                # Create a dataframe and save to CSV
                if texts:  # Only create a CSV if there are results
                    df = pd.DataFrame({
                        "Text": texts,
                        "Boundary Box Coordinates": bbox_coords,
                        "Confidence (%)": confidences
                    })
                    csv_path = os.path.join(process_folder, f"Results_{os.path.splitext(filename)[0]}.csv")
                    df.to_csv(csv_path, index=False)

print(f"Processed all images. Check the '{output_base_folder}' folder for results.")
