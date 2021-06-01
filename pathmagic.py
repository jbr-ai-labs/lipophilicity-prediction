import inspect
import os
import sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(current_dir, 'scripts'))
sys.path.insert(0, os.path.join(current_dir, 'scripts', 'preprocess'))
sys.path.insert(0, os.path.join(current_dir, 'scripts', 'SOTA/dmpnn/chemprop/train'))