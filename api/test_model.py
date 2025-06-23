# test_model.py - Testing script
from model_predictor import TableClassifier

def test_model():
    """
    Test the trained model with sample images
    """
    import os
    import glob
    
    # Initialize classifier
    classifier = TableClassifier(
        model_path="table_classifier.h5",
        class_indices_path="class_indices.json"
    )
    
    print("=== Model Testing ===")
    print(f"Classes: {list(classifier.class_indices.keys())}")
    print(f"Model input shape: {classifier.input_shape}")
    print()
    
    # Test with sample images
    test_images_dir = "test_images"  # Create this directory with test images
    
    if os.path.exists(test_images_dir):
        image_files = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                     glob.glob(os.path.join(test_images_dir, "*.png")) + \
                     glob.glob(os.path.join(test_images_dir, "*.jpeg"))
        
        if image_files:
            for image_path in image_files[:5]:  # Test first 5 images
                print(f"Testing: {os.path.basename(image_path)}")
                try:
                    result = classifier.predict(image_path)
                    print(f"  Top prediction: {result['top_prediction']['class']} "
                          f"({result['top_prediction']['confidence']:.3f})")
                    print(f"  All predictions:")
                    for pred in result['predictions']:
                        print(f"    - {pred['class']}: {pred['confidence']:.3f}")
                    print()
                except Exception as e:
                    print(f"  Error: {e}")
                    print()
        else:
            print(f"No image files found in {test_images_dir}")
    else:
        print(f"Test directory {test_images_dir} not found")
        print("Create a 'test_images' directory with sample images to test")

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Test the model
    test_model()
    
    # Or run the API server
    # run_api()