import pytesseract
from PIL import Image
import json
import os

# Configure pytesseract path
# Purpose: Ensure pytesseract is correctly configured to locate the OCR executable.
def configure_tesseract(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Image-to-Text Conversion
# Purpose: Extract text from an image using Tesseract OCR.
# Input: Image file path.
# Output: Extracted text.
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Batch Processing of Images
# Purpose: Extract text from all images in a directory.
# Input: Directory path containing image files.
# Output: Dictionary mapping image file names to extracted text.
def process_images_in_directory(image_dir):
    extracted_data = {}
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            text = extract_text_from_image(image_path)
            if text:
                extracted_data[image_file] = text
    return extracted_data

# Save Extracted Data
# Purpose: Save the extracted text data to a JSON file.
# Input: Extracted data (dictionary) and output file path.
# Output: JSON file containing the extracted data.
def save_extracted_data_to_json(extracted_data, output_file):
    try:
        with open(output_file, 'w') as json_file:
            json.dump(extracted_data, json_file, indent=4)
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

# Main Execution Pipeline
if __name__ == "__main__":
    # Configure Tesseract executable path
    tesseract_path = "/usr/bin/tesseract"  # Update path as necessary
    configure_tesseract(tesseract_path)

    # Directory containing images
    image_dir = "path_to_images"  # Replace with actual directory path

    # Extract text from images
    extracted_data = process_images_in_directory(image_dir)

    # Save extracted data to JSON
    output_file = "extracted_text_data.json"
    save_extracted_data_to_json(extracted_data, output_file)

    print(f"Text extraction completed. Results saved to {output_file}.")
