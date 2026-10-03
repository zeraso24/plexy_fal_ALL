import importlib
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Folder with the JS extension (shows/hides widgets depending on the selected model)
WEB_DIRECTORY = "./web"

MODULES = [
    "PLEXARD_FAL_multi_image",
]

for module_name in MODULES:
    try:
        imported_module = importlib.import_module(f".{module_name}", __name__)
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {
            **NODE_DISPLAY_NAME_MAPPINGS,
            **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
        }
        print(f"✓ {module_name}: Loaded {len(imported_module.NODE_CLASS_MAPPINGS)} node(s)")
    except Exception as e:
        print(f"✗ {module_name}: Failed to load nodes")
        print(f"  Error: {str(e)}")
        traceback.print_exc()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
