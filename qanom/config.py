import os, sys
_current_module = sys.modules[__name__]
qanom_package_root_path = os.path.dirname(os.path.abspath(_current_module.__file__))
resources_path = os.path.join(qanom_package_root_path, 'resources')