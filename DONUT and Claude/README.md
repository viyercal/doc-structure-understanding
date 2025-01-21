Document Structure Extraction Methodology: DONUT and Claude
This project utilizes a two-pronged approach involving the DONUT model and the Claude large language model to extract structured information from document images. The methodology can be broadly divided into two sections:

I. DONUT Model for Initial Structure Prediction
Overview
The DONUT (Document Understanding Transformer) model is employed as the first stage to generate an initial prediction of the document's structure. DONUT is an OCR-free model that excels at understanding document layouts and extracting information without relying on traditional OCR engines.

Steps
Image Preprocessing:

Input document images (in PNG format) are resized to a standardized dimension (960x1280 in this case) to ensure consistent input to the DONUT model.

Images are converted to RGB format to ensure compatibility with the processor.

Tokenization and Feature Extraction:

The DonutProcessor is used to prepare the input images for the DONUT model. This involves:

Converting the image into a sequence of "patches" (smaller image segments).

Generating pixel_values, which are numerical representations of the image patches.

Inference with DONUT:

The pixel_values are fed into the DONUT model (VisionEncoderDecoderModel).

The model generates a sequence of tokens representing the predicted structure of the document. This sequence uses special tags (defined in the "Tag Set" section below) to denote different document elements.

The generation process utilizes techniques like beam search to improve the quality of the predicted sequence.

Output Processing:

The generated token sequence is decoded using the DonutProcessor to obtain a human-readable string.

Special tokens (e.g., <s>, </s>, <pad>) are removed to clean up the output.

Tag Set
The DONUT model is trained to recognize and output the following tags:

paragraph

page_header

page_footer

title

subheading

table

image

code_snippet

These tags are used to demarcate different structural elements within the document.

II. Claude for Refinement and Detailed Extraction
Overview
The output from the DONUT model serves as a starting point for further refinement and more detailed information extraction using Claude, a large language model from Anthropic. Claude is provided with the original document image and the DONUT model's output to generate a more accurate and comprehensive representation of the document's structure.

Steps
Image Preparation:

The original document image (PNG format) is converted into a base64 encoded string. This format is suitable for inputting images to the Claude API.

Prompt Engineering:

A carefully crafted prompt is used to instruct Claude on its task. The prompt includes:

Role Definition: Defines Claude as a "document structure expert."

Task Instructions: Explicitly instructs Claude to identify the document structure, delineate paragraphs, and output in a specific format.

Output Format: Specifies that the output should use the same tags as the DONUT model and always end with the <s> tag.

Example: Provides example inputs and outputs to guide Claude's understanding of the desired format.

Inference with Claude:

The base64 encoded image and the prompt are sent to the Claude API.

Claude processes the image and the instructions to generate a text response representing its interpretation of the document structure.

Output:

Claude's output is a text string containing tags that delineate the document's elements, similar to the DONUT model's output but potentially more refined and detailed.

Tag Set (Claude)
Claude is instructed to use a more extensive set of tags, including:

paragraph

page_number

image

paragraphs_in_image

title

table

paragraphs_in_table

other

page_header

subheading

formulas

page_footer

paragraphs_in_form

checkbox

checkbox_checked

form

radio_button_checked

radio_button

code_snippet

III. Data Parsing and Accuracy Calculation
Conceptual Data Parsing
XML to Dictionary (Intermediate Step):

Initially, the ground truth data is assumed to be in XML format.

An intermediate step (used during training data creation) involves converting the XML files into a list of dictionaries. Each dictionary represents a tagged element within the document (e.g., {"paragraph": "some text"}).

JSON Representation:

The parsed data (from both DONUT and Claude outputs, and ground truth) is represented in JSON format. Each JSON object contains:

file_name: The name of the corresponding image file.

text: A list of dictionaries, where each dictionary represents a tagged element (e.g., {"paragraph": "some text"}). This format is designed to be easily processed by the DONUT model during training.

For evaluation, the 'text' element of the JSON object can be compared between the prediction and the ground truth.

Accuracy Calculation
The accuracy of the document structure extraction is evaluated using several metrics:

Overall Text Extraction Accuracy:

Calculated using the Levenshtein distance to compare the extracted text content with the ground truth text content.

Measures the overall accuracy of text extraction across all document elements.

Tag Categorization Accuracy:

Measures the accuracy of correctly assigning tags to document elements.

Calculated as the ratio of correctly classified tags to the total number of tags.

Provides both an overall tag accuracy and tag-specific accuracies (e.g., accuracy for "paragraph" tags, "title" tags, etc.).

Text Extraction Accuracy by Tag:

Calculates the text extraction accuracy for each specific tag (e.g., "paragraph", "title", "image").

Uses the Levenshtein distance to compare the extracted text for a given tag with the corresponding ground truth text.

Provides a more granular view of the model's performance on different document elements.

Special Handling in Accuracy Calculation:
*   **Missing Tags:** If a tag is missing in both the reference and prediction, it's not considered in the accuracy calculation (marked as -9999 internally) to avoid penalizing the model for tags that are not present in the ground truth.
*   **Extra Predictions:** If the prediction contains extra text elements not present in the reference, these are also considered in the accuracy calculations to penalize hallucinations.
Use code with caution.
These metrics provide a comprehensive evaluation of the model's ability to both extract text accurately and correctly identify the structure of the document.
