# a python script for qasrl-crowdsourcing project, to run the qanom (python) script for 
# identifying candidate nominalization for sentences.
# This script can be called from everywhere, and it will run the qanom module 'prepare_qanom_prompts.py' after changing directroy as required, using a python executable in 'QANom/.venv3'.

import os, sys, subprocess

# globals
qanom_repo_root_locaction = 'QANom' 	# location of the QANom project root directory 
							# (where the qanom package and virtual env resides)
assert os.path.isdir(qanom_repo_root_locaction), "Wrong location for QANom repository"							
							
qanom_repo_root = os.path.abspath(qanom_repo_root_locaction)							
virtual_env_name = '.venv3'							
python_exe = os.path.join(qanom_repo_root, virtual_env_name, 'bin/python')
assert os.path.isfile(python_exe), "Cannot find python executable"

main_package = 'qanom'
main_package_path = os.path.join(qanom_repo_root, main_package)
script_name = 'prepare_qanom_prompts.py'
script = os.path.join(qanom_repo_root, main_package, script_name)
assert os.path.isfile(script), "Cannot find script: " + script 

# cmd-line args
source_fn, out_fn = sys.argv[-2:] 
assert os.path.isfile(source_fn), "Cannot find source-sentences file: " + source_fn
source_abs_path = os.path.abspath(source_fn)
out_abs_path = os.path.abspath(out_fn)

# command for execution
cmd = "{} {} {} {}".format(python_exe, script, source_abs_path, out_abs_path)

# execute:
cwd = os.getcwd()
os.chdir(qanom_repo_root)
subprocess.call(cmd, shell=True)	# actual module execution
os.chdir(cwd)
