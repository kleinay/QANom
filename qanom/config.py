import os, sys
_current_module = sys.modules[__name__]
current_path = os.path.dirname(os.path.abspath(_current_module.__file__))
repository_root_path = os.path.dirname(current_path)
resources_path = os.path.join(repository_root_path, 'resources')
