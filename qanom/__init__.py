import os
package_path = os.path.dirname(os.path.abspath(__file__)) # 'qanom' package base directory, e.g. "/home/.../site-packages/qanom" 
with open(os.path.join(package_path, "version.txt")) as f:
    __version__ = f.read().strip()