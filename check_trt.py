import sys
print(f"Python: {sys.version}")
try:
    import tensorrt
    print("✅ SUCCESS: TensorRT is found!")
    print(f"Version: {tensorrt.__version__}")
except ImportError as e:
    print(f"❌ FAILURE: {e}")
    print("Search paths:", sys.path)