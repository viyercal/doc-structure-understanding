# DONUT, Claude, Accuracy
We tested DONUT and Claude on extracting these tags and the information contained within: paragraph, image, title, table, page_header, subheading, code_snippet, page_footer.

## DONUT:
1. Using PyTesseract, created XMLs from PNGs and Bounding Box Data in COCO format.
2. Create JSONs of the XMLs, key being the tag, value being the text
3. Convert JSONs to DONUT readable format
4. Convert Images to RGB format pixels and resize to 960x1280
5. Create Train, Test, and Eval Datasets

## Claude:
1. Convert RGB format pixels to PNG for Claude
2. Create Prompt to ensure it only outputs certain tags and format

## Accuracy:
### Overall Text Extraction Accuracy:
Calculated using the Levenshtein distance to compare the predicted text content with the ground truth text content.
Measures the overall accuracy of text extraction across all document elements.

### Tag Categorization Accuracy:
Measures the accuracy of correctly assigning tags to document elements by calculating the ratio of correctly classified tags to the total number of tags for each tag

### Overall Tag Categorization Accuracy:
Measures the accuracy of correctly assigning tags to document elements by calculating the ratio of correctly classified tags to the total number of tags in general.

### Text Extraction Accuracy by Tag:
Calculates the text extraction accuracy for each specific tag using the Levenshtein distance to compare the extracted text for a given tag with the corresponding ground truth text.

### Overall Tag Categorization F1 Score:
Measures the accuracy of correctly assigning tags to document elements by calculating the F1 score.

### Tag-Specific F1 Scores:
Measures the accuracy of correctly assigning tags to each specific document elements by calculating the F1 score.

#### Extra text elements not present in the reference are penalized (hallucinations)
