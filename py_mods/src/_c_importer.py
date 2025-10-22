import sys

def get_lib_ext() -> str:
    if sys.platform.startswith("darwin"):
        return ".dylib"
    elif sys.platform.startswith("linux"):
        return ".so"
    elif sys.platform.startswith("win"):
        return ".dll"
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
