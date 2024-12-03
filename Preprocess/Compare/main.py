import easyocr
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from jiwer import wer
# !pip install jiwer

def process_easyocr(image_path):
    reader = easyocr.Reader(['tr'])  # Turkish language
    results = reader.readtext(image_path, detail=1, paragraph=False)
    return results


def save_image_with_text(image_path, results, output_image_path, method='easyocr'):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    if method == 'easyocr':
        for result in results:
            (start_x, start_y), (end_x, end_y) = result[0][0], result[0][2]
            text = result[1]
            # Draw text above the bounding box
            text_position = (start_x, start_y - 10)  # 10 pixels above the rectangle
            draw.text(text_position, text, fill='red')
    
    # Save the image with text
    image.save(output_image_path)

def save_easyocr_results_to_csv(easyocr_results, csv_path):
    easyocr_data = []
    for result in easyocr_results:
        text = result[1]
        confidence = result[2]  # EasyOCR confidence score
        easyocr_data.append([text, f"{confidence:.2f}"])  # Format confidence to 2 decimal places
    
    # Save EasyOCR results to CSV
    df = pd.DataFrame(easyocr_data, columns=['Text', 'Confidence'])
    df.to_csv(csv_path, index=False)


def calculate_wer(csv_file_path, ground_truth_path, conf_level=0.7):
    """
    Calculate the Word Error Rate (WER) for the given OCR CSV file against the ground truth.

    Parameters:
    csv_file_path (str): Path to the CSV file containing OCR results (Text, Confidence).
    ground_truth_path (str): Path to the ground truth text file.
    conf_level (float): Confidence threshold for filtering OCR results.

    Returns:
    float: WER value for the given CSV file.
    """
    # Load OCR results
    ocr_df = pd.read_csv(csv_file_path)
    
    # Filter by confidence level
    filtered_ocr_df = ocr_df[ocr_df['Confidence'] >= conf_level]
    
    # Combine filtered OCR results into one string
    ocr_text = " ".join(filtered_ocr_df['Text'].astype(str))

    # Load ground truth words
    with open(ground_truth_path, "r", encoding="utf-8") as gt_file:
        ground_truth_text = gt_file.read().strip()

    # Calculate WER
    wer_score = wer(ground_truth_text, ocr_text)
    return wer_score

def create_wer_table(    
                    easyocr_csv_path = r"C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\easyocr_results.csv",
                    ground_truth_path = r"C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\words.txt",
                    output_csv_path = r"C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\wer_results.csv",
                    ):

    # Calculate WER for confidence levels from 0.0 to 1.0 with a step of 0.1
    confidence_levels = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]
    results = []

    for conf in confidence_levels:
        easyocr_wer = calculate_wer(easyocr_csv_path, ground_truth_path, conf_level=conf)
        results.append({'Confidence Level': conf, 'EasyOCR WER': easyocr_wer})

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_csv_path, index=False)

    print(f"WER results saved to: {output_csv_path}")
    return results_df

def main():
    image_path = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Preprocess\Masked Images\masked_image.png'
    output_image_easyocr = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Preprocess\Compare\easyocr_output.png'
    csv_output_easyocr_path = r'C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Preprocess\Compare\easyocr_results.csv'
    
    # Process the image with EasyOCR
    easyocr_results = process_easyocr(image_path)
    
    # Save images with text only (no bounding boxes)
    save_image_with_text(image_path, easyocr_results, output_image_easyocr, method='easyocr')
    
    # Save results to separate CSV files
    save_easyocr_results_to_csv(easyocr_results, csv_output_easyocr_path)

    print("Processing complete! Images and CSV files saved.")

    # Creat a csv file that has comparison of two algortihm
    create_wer_table(
                    easyocr_csv_path = csv_output_easyocr_path,
                    ground_truth_path = r"C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Preprocess\Compare\words.txt",
                    output_csv_path = r"C:\Users\AZMI\Desktop\Izzet Ahmet\Kodlar\OCR\Preprocess\Compare\wer_results.csv",)



# Run the main function
if __name__ == "__main__":
    main()
