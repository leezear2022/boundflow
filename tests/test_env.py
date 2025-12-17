import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not Set')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
if os.environ.get("CONDA_DEFAULT_ENV") != "boundflow":
    print("[HINT] 建议在 conda 环境 'boundflow' 下运行：`conda activate boundflow && python tests/test_env.py`")
print("-" * 40)

def test_import(name, package_name=None):
    if package_name is None: 
        package_name = name
    try:
        __import__(package_name)
        print(f"[OK] Import {name} success")
        return True
    except ImportError as e:
        print(f"[FAIL] Import {name} failed: {e}")
        return False
    except Exception as e:
        print(f"[WARN] Import {name} error (compiled lib missing?): {e}")
        return False

success = True
success &= test_import("PyTorch", "torch")
success &= test_import("Auto_LiRPA", "auto_LiRPA")
success &= test_import("BoundFlow", "boundflow")
# TVM often requires libtvm.so to be built. We expect python import to find the package 
# but potentially fail on loading the shared library if not built.
# We treat specific LoadErrors as "Partial Success" (path correct, build needed).
try:
    import tvm
    print("[OK] Import TVM success")
except ImportError as e:
     print(f"[FAIL] Import TVM failed: {e}")
     success = False
except Exception as e:
    print(f"[WARN] Import TVM partial success (libtvm missing?): {e}")
    # This is expected before compilation
    pass

if success:
    print("-" * 40)
    print("Environment verification passed!")
    sys.exit(0)
else:
    print("-" * 40)
    print("Environment verification FAILED.")
    sys.exit(1)
