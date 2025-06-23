import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging
from pathlib import Path

class ImagePreprocessor:
    """Advanced image preprocessing for handwritten table detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            # Convert BGR to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def denoise_image(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Remove noise while preserving edges"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'nlm':
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            return image
    
    def correct_perspective(self, image: np.ndarray, corners: Optional[np.ndarray] = None) -> np.ndarray:
        """Correct perspective distortion using detected corners or automatic detection"""
        if corners is None:
            corners = self._detect_document_corners(image)
            
        if corners is not None and len(corners) == 4:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(corners)
            
            # Calculate dimensions of the corrected image
            width = max(
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[3])
            )
            height = max(
                np.linalg.norm(corners[3] - corners[0]),
                np.linalg.norm(corners[2] - corners[1])
            )
            
            # Define destination points
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transform matrix
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
            
            # Apply perspective correction
            corrected = cv2.warpPerspective(image, matrix, (int(width), int(height)))
            return corrected
        
        return image
    
    def _detect_document_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect document corners using contour detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assumed to be the document)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Return corners if we found a quadrilateral
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        # Calculate centroid
        center = np.mean(corners, axis=0)
        
        # Sort by angle from centroid
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Find top-left (smallest sum of coordinates)
        sums = [corner[0] + corner[1] for corner in sorted_corners]
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        ordered = sorted_corners[top_left_idx:] + sorted_corners[:top_left_idx]
        
        return np.array(ordered)
    
    def preprocess_image(self, image: np.ndarray, 
                        enhance_contrast: bool = True,
                        denoise: bool = True,
                        correct_perspective: bool = True,
                        denoise_method: str = 'bilateral') -> np.ndarray:
        """Complete preprocessing pipeline"""
        processed = image.copy()
        
        if enhance_contrast:
            processed = self.enhance_contrast(processed)
            self.logger.info("Applied contrast enhancement")
        
        if denoise:
            processed = self.denoise_image(processed, method=denoise_method)
            self.logger.info(f"Applied denoising with {denoise_method} method")
        
        if correct_perspective:
            processed = self.correct_perspective(processed)
            self.logger.info("Applied perspective correction")
        
        return processed


class TableDetector:
    """YOLOv8-based table detection with confidence scoring"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded custom YOLOv8 model from {model_path}")
        else:
            # Use pretrained model - you'll need to fine-tune this for table detection
            self.model = YOLO('yolov8n.pt')  # Start with nano model
            self.logger.warning("Using pretrained YOLOv8 - recommend fine-tuning for table detection")
    
    def detect_tables(self, image: np.ndarray, 
                     confidence_threshold: float = 0.5,
                     iou_threshold: float = 0.4) -> List[dict]:
        """Detect tables in the image and return bounding boxes with confidence scores"""
        
        # Run inference
        results = self.model(image, conf=confidence_threshold, iou=iou_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.logger.info(f"Detected {len(detections)} tables")
        return detections
    
    def extract_table_regions(self, image: np.ndarray, detections: List[dict], 
                            padding: int = 10) -> List[Tuple[np.ndarray, dict]]:
        """Extract table regions from image based on detections"""
        table_regions = []
        
        h, w = image.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Add padding while staying within image bounds
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract region
            table_region = image[y1:y2, x1:x2]
            
            # Update detection with padded coordinates
            detection_copy = detection.copy()
            detection_copy['bbox'] = [x1, y1, x2, y2]
            
            table_regions.append((table_region, detection_copy))
        
        return table_regions


class TablePreprocessingPipeline:
    """Complete pipeline for Step 1: Image Preprocessing & Table Detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.preprocessor = ImagePreprocessor()
        self.detector = TableDetector(model_path)
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
    
    def process_image(self, image_path: str, 
                     preprocess_config: dict = None,
                     detection_config: dict = None) -> dict:
        """
        Complete processing pipeline for a single image
        
        Args:
            image_path: Path to input image
            preprocess_config: Preprocessing configuration
            detection_config: Detection configuration
            
        Returns:
            Dictionary containing results
        """
        
        # Default configurations
        if preprocess_config is None:
            preprocess_config = {
                'enhance_contrast': True,
                'denoise': True,
                'correct_perspective': True,
                'denoise_method': 'bilateral'
            }
        
        if detection_config is None:
            detection_config = {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.4,
                'padding': 10
            }
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.logger.info(f"Processing image: {image_path}")
        self.logger.info(f"Original image shape: {image.shape}")
        
        # Step 1: Preprocess image
        preprocessed_image = self.preprocessor.preprocess_image(
            image, **preprocess_config
        )
        cv2.imwrite("preprocessed.jpg", preprocessed_image)
        
        # Step 2: Detect tables
        detections = self.detector.detect_tables(
            preprocessed_image,
            confidence_threshold=detection_config['confidence_threshold'],
            iou_threshold=detection_config['iou_threshold']
        )
        
        # Step 3: Extract table regions
        table_regions = self.detector.extract_table_regions(
            preprocessed_image, 
            detections, 
            padding=detection_config['padding']
        )
        
        # Prepare results
        results = {
            'original_image': image,
            'preprocessed_image': preprocessed_image,
            'detections': detections,
            'table_regions': table_regions,
            'num_tables': len(detections),
            'image_shape': preprocessed_image.shape
        }
        
        self.logger.info(f"Processing complete. Found {len(detections)} tables.")
        
        return results
    
    def visualize_results(self, results: dict, save_path: Optional[str] = None) -> np.ndarray:
        """Visualize detection results"""
        image = results['preprocessed_image'].copy()
        
        # Draw bounding boxes
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"Table: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"Visualization saved to {save_path}")
        
        return image


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TablePreprocessingPipeline()
    
    # Process an image
    try:
        results = pipeline.process_image(
            "/Users/daniilkaimekin/Projects/nurik_forensics/dl/media/IMG_0792.JPG",
            preprocess_config={
                'enhance_contrast': True,
                'denoise': True,
                'correct_perspective': True,
                'denoise_method': 'bilateral'
            },
            detection_config={
                'confidence_threshold': 0.2,  # Lower threshold for better recall
                'iou_threshold': 0.4,
                'padding': 15
            }
        )
        
        # Visualize results
        visualization = pipeline.visualize_results(results, "detected_tables.jpg")
        
        # Print summary
        print(f"Processing Summary:")
        print(f"- Original image shape: {results['original_image'].shape}")
        print(f"- Preprocessed image shape: {results['preprocessed_image'].shape}")
        print(f"- Number of tables detected: {results['num_tables']}")
        
        for i, (table_region, detection) in enumerate(results['table_regions']):
            print(f"- Table {i+1}: {table_region.shape}, confidence: {detection['confidence']:.3f}")
            
    except Exception as e:
        print(f"Error processing image: {e}")