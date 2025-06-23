# @title Table Recognition for Kazakh/Russian (Cyrillic) Documents
# Install dependencies
# !sudo apt install tesseract-ocr-kaz tesseract-ocr-rus
# !pip install pytesseract opencv-python pandas transformers torch


import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
from google.colab.patches import cv2_imshow  # For displaying images in Colab

# --------------------------
# STEP 1: Load and Preprocess Image
# --------------------------
def load_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return img, thresh

# --------------------------
# STEP 2: Detect Table Structure (OpenCV)
# --------------------------
def detect_table_structure(thresh_img):
    # Detect lines
    kernel_length = np.array(thresh_img).shape[1] // 80
    horz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    
    horz_lines = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, horz_kernel, iterations=3)
    vert_lines = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, vert_kernel, iterations=3)
    
    # Combine lines
    grid = cv2.addWeighted(horz_lines, 0.5, vert_lines, 0.5, 0.0)
    grid = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Find contours
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --------------------------
# STEP 3: OCR with Kazakh/Russian Support
# --------------------------
def extract_text_from_cell(cell_img):
    # Use Tesseract with Kazakh+Russian
    text = pytesseract.image_to_string(
        cell_img, 
        lang='kaz+rus', 
        config='--psm 6 --oem 1'  # PSM 6 = Assume uniform block
    )
    return clean_cyrillic_text(text)

# --------------------------
# STEP 4: Clean Cyrillic Text (Kazakh/Russian)
# --------------------------
def clean_cyrillic_text(text):
    # Kazakh/Russian character ranges + common symbols
    cyrillic_chars = r'[^а-яА-ЯәғқңөұүіһӘҒҚҢӨҰҮІҐ0-9\s\.,\-–—]'
    text = re.sub(cyrillic_chars, '', text)
    
    # Common OCR misreads in Cyrillic
    corrections = {
        "с": "с", "ѕ": "с",  # Tesseract confuses these
        "Њ": "Ң", "њ": "ң",  # Kazakh-specific fixes
        "o": "о", "O": "О"   # Latin vs Cyrillic 'o'
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text.strip()

# --------------------------
# STEP 5: Convert Cells to CSV
# --------------------------
def cells_to_dataframe(cells):
    # Sort cells by (y, x) coordinates to reconstruct rows
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    
    # Group into rows (tolerance for small y-variations)
    rows = []
    current_row = []
    prev_y = None
    
    for x, y, text in cells:
        if prev_y is None or abs(y - prev_y) < 10:  # Same row
            current_row.append(text)
        else:
            rows.append(current_row)
            current_row = [text]
        prev_y = y
    
    if current_row:  # Add last row
        rows.append(current_row)
    
    return pd.DataFrame(rows)

# --------------------------
# MAIN PIPELINE
# --------------------------
def table_to_csv(image_path, output_csv='output.csv'):
    # 1. Load image
    img, thresh = load_image(image_path)
    
    # 2. Detect table structure
    contours = detect_table_structure(thresh)
    
    # 3. Extract text from each cell
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 20:  # Skip small artifacts
            continue
        cell_img = img[y:y+h, x:x+w]
        text = extract_text_from_cell(cell_img)
        cells.append((x, y, text))
    
    # 4. Convert to DataFrame and save
    df = cells_to_dataframe(cells)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"CSV saved to {output_csv}")
    return df

# --------------------------
# RUN ON SAMPLE IMAGE (Upload your own!)
# --------------------------
# from google.colab import files
# uploaded = files.upload()
# image_path = list(uploaded.keys())[0]

image_path = '/content/drive/MyDrive/data/nurik/Screenshot 2025-06-21 at 17.00.35.png'

# Process and show results
df = table_to_csv(image_path)
display(df.head())


from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load pre-trained model for handwritten Cyrillic
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def extract_handwritten_text(cell_img):
    # Convert OpenCV image (BGR) to RGB
    cell_img_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
    pixel_values = processor(cell_img_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def preprocess_handwritten(img):
    # Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Binarization (adaptive for variable lighting)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    return thresh



from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor

feature_extractor = DetrFeatureExtractor()
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

def detect_table_dl(img):
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    return outputs  # Process outputs to get cell coordinates


def clean_handwriting(text):
    # Merge broken letters (e.g., "к а з" → "каз")
    text = re.sub(r'([а-яәғқңөұүіһ])\s+([а-яәғқңөұүіһ])', r'\1\2', text)
    # Fix common Cyrillic misreads
    corrections = {"и": "й", "п": "р", "c": "с"}  # Add more based on your data
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text



import torch
### 1. Border Detection Function
def has_visible_borders(img, threshold=0.02):
    """
    Detect if image has visible table borders using line detection.
    Returns True if significant horizontal/vertical lines are found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=img.shape[1]//3, maxLineGap=10)
    
    if lines is None:
        return False
    
    # Count significant horizontal/vertical lines
    h_lines, v_lines = 0, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 5:  # Horizontal line
            h_lines += 1
        elif abs(x2 - x1) < 5:  # Vertical line
            v_lines += 1
    
    # Check if we have enough lines to form a table
    min_lines = max(img.shape[0], img.shape[1]) * threshold
    return (h_lines >= min_lines) and (v_lines >= min_lines)

### 2. Deep Learning Output Processing
def process_dl_outputs(outputs, img, confidence_threshold=0.8):
    """
    Process Table Transformer outputs to extract cell coordinates.
    Returns list of contours for detected cells.
    """
    # Convert model outputs to usable format
    target_sizes = torch.tensor([img.shape[:2]])
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]
    
    # Convert detections to OpenCV contours
    contours = []
    for box in results["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        contour = np.array([[
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ]], dtype=np.int32)
        contours.append(contour)
    
    return contours

### 3. Updated Main Pipeline
def handwritten_table_to_csv(image_path):
    # Initialize TrOCR processor (load once)
    if 'processor' not in globals():
        global processor, model
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # 1. Preprocess
    img = cv2.imread(image_path)
    processed_img = preprocess_handwritten(img)
    
    # 2. Detect table structure
    if has_visible_borders(img):
        # Traditional approach for bordered tables
        contours = detect_table_structure(processed_img)
    else:
        # Deep learning for borderless tables
        inputs = feature_extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        contours = process_dl_outputs(outputs, img)
    
    # 3. Extract text from each cell
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cell_img = img[y:y+h, x:x+w]
        
        # Skip very small detections (likely noise)
        if w < 10 or h < 10:
            continue
            
        text = extract_handwritten_text(cell_img)
        text = clean_handwriting(text)
        cells.append((x, y, text))
    
    # 4. Convert to structured DataFrame
    df = cells_to_dataframe(cells)
    output_path = "output_handwritten.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_path}")
    return df

### 4. Helper Functions (Previously Defined)
def preprocess_handwritten(img):
    """Enhance handwritten text visibility"""
    # Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def extract_handwritten_text(cell_img):
    """Extract text using TrOCR"""
    cell_img_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
    pixel_values = processor(cell_img_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def clean_handwriting(text):
    """Fix common handwriting recognition errors"""
    # Merge broken characters
    text = re.sub(r'([а-яәғқңөұүіһӘҒҚҢӨҰҮІҐ])\s+([а-яәғқңөұүіһӘҒҚҢӨҰҮІҐ])', r'\1\2', text)
    # Common misreads in Cyrillic handwriting
    corrections = {
        "и": "й", "п": "р", "c": "с", "т": "г", 
        "е": "ә", "o": "ө", "к": "қ", "н": "ң"
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text.strip()

def cells_to_dataframe(cells):
    """Convert (x,y,text) cells to structured DataFrame"""
    # Sort by Y then X to get row-major order
    cells.sort(key=lambda c: (c[1], c[0]))
    
    # Group into rows (all cells with similar Y coordinates)
    rows = []
    current_row = []
    prev_y = None
    
    for x, y, text in cells:
        if prev_y is None or abs(y - prev_y) < 15:  # Same row
            current_row.append((x, text))
        else:
            # Sort current row by X and extract texts
            current_row.sort()
            rows.append([text for _, text in current_row])
            current_row = [(x, text)]
        prev_y = y
    
    # Add the last row
    if current_row:
        current_row.sort()
        rows.append([text for _, text in current_row])
    
    return pd.DataFrame(rows)

# Example usage
if __name__ == "__main__":
    from google.colab import files
    uploaded = files.upload()
    image_path = '/content/drive/MyDrive/data/nurik/Screenshot 2025-06-21 at 17.00.35.png'
    result_df = handwritten_table_to_csv(image_path)
    display(result_df)

