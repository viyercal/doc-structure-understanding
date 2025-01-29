import pytesseract
from PIL import Image
import json
import os

def configure_tesseract(tesseract_path):
    """
    Purpose: Configure the Tesseract OCR executable path.
    Input: tesseract_path (str): Path to the Tesseract executable.
    Output: None.
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def extract_text_from_image(image_path):
    """
    Purpose: Extract text from an image file.
    Input: image_path (str): Path to the image file.
    Output: Extracted text (str).
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

def process_images_in_directory(image_dir):
    """
    Purpose: Extract text from all images in a specified directory.
    Input: image_dir (str): Path to the directory containing images.
    Output: extracted_data (dict): Dictionary mapping image file names to extracted text.
    """
    extracted_data = {}
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            text = extract_text_from_image(image_path)
            extracted_data[image_file] = text
    return extracted_data

def save_extracted_data_to_json(extracted_data, output_file):
    """
    Purpose: Save extracted text data to a JSON file.
    Input: extracted_data (dict): Dictionary containing text data.
           output_file (str): Path to the output JSON file.
    Output: None.
    """
    try:
        with open(output_file, 'w') as json_file:
            json.dump(extracted_data, json_file, indent=4)
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")


if __name__ == "__main__":
    """
    Execute the OCR pipeline: configure Tesseract, process images, and save extracted text.
    """
    # Configure Tesseract executable path
    tesseract_path = "/usr/bin/tesseract"  # Update this path as needed
    configure_tesseract(tesseract_path)

    # Directory containing images
    image_dir = "path_to_images"  # Replace with the actual directory path

    # Extract text from images
    extracted_data = process_images_in_directory(image_dir)

    # Save extracted text to a JSON file
    output_file = "extracted_text_data.json"
    save_extracted_data_to_json(extracted_data, output_file)

    print(f"Text extraction completed. Results saved to {output_file}.")
