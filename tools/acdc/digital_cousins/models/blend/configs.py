import os

# Use environment variable or default to a relative path from project root
_PROJECT_ROOT = os.environ.get("ACDC_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
RENDER_DIR = os.environ.get("ACDC_RENDER_DIR", os.path.join(_PROJECT_ROOT, "output", "render") + "/")