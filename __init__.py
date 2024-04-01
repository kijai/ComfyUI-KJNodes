import subprocess
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
