import os

# Check if file exists
model_path = 'ACCESS_MODELS/SVM/best_svm_pipeline.pkl'
if not os.path.exists(model_path):
    print(f"File {model_path} does not exist")
    exit(1)

# Check file size
size = os.path.getsize(model_path)
print(f"File size: {size} bytes")

# Read first few bytes to check header
with open(model_path, 'rb') as f:
    header = f.read(10)
    print("File header (hex):", header.hex())
    print("File header (raw):", header)
