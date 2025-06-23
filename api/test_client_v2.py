# test_client.py - Test the API
import requests
import json
import os
from pathlib import Path

def test_api():
    """Test the Table Classifier API"""
    base_url = "http://localhost:8000"
    
    print("=== Testing Table Classifier API ===\n")
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return
    print()
    
    # Test 2: Get classes
    print("2. Getting available classes...")
    try:
        response = requests.get(f"{base_url}/classes")
        classes_data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Classes: {classes_data['classes']}")
        print(f"Number of classes: {classes_data['num_classes']}")
        print(f"Classes with tables: {classes_data.get('classes_with_tables', [])}")
        print(f"Classes without tables: {classes_data.get('classes_without_tables', [])}")
    except Exception as e:
        print(f"Error: {e}")
        return
    print()
    
    # Test 2.5: Get available tables
    print("2.5. Getting available tables...")
    try:
        response = requests.get(f"{base_url}/tables")
        tables_data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Available tables: {tables_data['available_tables']}")
        print(f"Tables count: {tables_data['count']}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Test 3: Detailed health check
    print("3. Detailed health check...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Health: {json.dumps(health_data, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Test 4: Single image prediction with table data
    print("4. Testing single image prediction with table data...")
    
    # Look for test images
    test_dir = Path("test_images")
    if test_dir.exists():
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg"))
        
        if image_files:
            test_image = image_files[0]
            print(f"Using test image: {test_image}")
            
            try:
                with open(test_image, 'rb') as f:
                    files = {'file': (test_image.name, f, 'image/jpeg')}
                    params = {'top_k': 3, 'include_table': True, 'table_format': 'dict'}
                    response = requests.post(f"{base_url}/predict", files=files, params=params)
                
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Filename: {result['filename']}")
                    print(f"Top prediction: {result['top_prediction']['class']} ({result['top_prediction']['confidence']:.3f})")
                    print("All predictions:")
                    for pred in result['predictions']:
                        print(f"  - {pred['class']}: {pred['confidence']:.3f}")
                    
                    # Show table data if available
                    if 'table_data' in result:
                        table_info = result['table_data']
                        if 'error' in table_info:
                            print(f"Table data: {table_info['error']}")
                        else:
                            print(f"Table data for '{table_info['table_name']}':")
                            print(f"  - Row count: {table_info.get('row_count', 'N/A')}")
                            if 'data' in table_info and table_info['data']:
                                print(f"  - Sample data (first row): {table_info['data'][0] if table_info['data'] else 'No data'}")
                else:
                    print(f"Error: {response.text}")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No image files found in test_images directory")
    else:
        print("test_images directory not found")
        print("Create a 'test_images' directory and add some images to test")
    print()
    
    # Test 4.5: Test direct table access
    print("4.5. Testing direct table access...")
    try:
        response = requests.get(f"{base_url}/tables")
        if response.status_code == 200:
            tables_data = response.json()
            available_tables = tables_data.get('available_tables', [])
            
            if available_tables:
                # Test getting data from first available table
                table_name = available_tables[0]
                print(f"Testing table: {table_name}")
                
                response = requests.get(f"{base_url}/tables/{table_name}")
                if response.status_code == 200:
                    table_data = response.json()
                    print(f"Table '{table_name}' loaded successfully")
                    print(f"  - Row count: {table_data['row_count']}")
                    if table_data['data']:
                        print(f"  - Sample row: {table_data['data'][0]}")
                else:
                    print(f"Error loading table: {response.text}")
            else:
                print("No tables available to test")
        else:
            print(f"Error getting tables list: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Test 5: Batch prediction (if multiple images available)
    print("5. Testing batch prediction...")
    if test_dir.exists():
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg"))
        
        if len(image_files) >= 2:
            test_images = image_files[:3]  # Test with up to 3 images
            print(f"Using {len(test_images)} test images")
            
            try:
                files = []
                for img_path in test_images:
                    with open(img_path, 'rb') as f:
                        files.append(('files', (img_path.name, f.read(), 'image/jpeg')))
                
                response = requests.post(f"{base_url}/predict_batch", files=files)
                
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Processed {len(result['results'])} images:")
                    for res in result['results']:
                        if res['status'] == 'success':
                            print(f"  {res['filename']}: {res['top_prediction']['class']} ({res['top_prediction']['confidence']:.3f})")
                        else:
                            print(f"  {res['filename']}: ERROR - {res['error']}")
                else:
                    print(f"Error: {response.text}")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Need at least 2 images for batch testing")
    print()
    
    print("=== API Testing Complete ===")

if __name__ == "__main__":
    test_api()