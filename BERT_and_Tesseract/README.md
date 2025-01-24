BERT + Tesseract: Document Structure Understanding

Overview

This repository focuses on the BERT + Tesseract pipeline for document structure understanding. The pipeline uses Tesseract OCR to extract raw text from document images and leverages BERT for hierarchical classification of document elements such as paragraphs, headers, and titles.
Key contributions include:
Integration of OCR-based preprocessing with BERT classification.
Evaluation of hierarchical metrics, such as text extraction accuracy and tag categorization accuracy.
Methodology

Pipeline Steps
OCR Preprocessing:
Tesseract processes document images to extract raw text.
Handles diverse formats like PDFs, PNGs, and JPEGs.
Text Tokenization:
Extracted text is tokenized using BERT tokenizer with positional encodings for hierarchical structure understanding.
Training:
Fine-tuned BERT to classify text into tags like <paragraph>, <title>, <page_header>, etc.
Evaluation:
Evaluated using metrics like accuracy, F1 score, and Levenshtein distance for hierarchical classification.