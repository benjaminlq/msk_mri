import importlib.util
import sys

def import_module_from_path(
    module_path: str, module_name: str  
):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    return module

