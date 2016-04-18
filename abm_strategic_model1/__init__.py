#from abm_strategic import import_exterior_libs
import os, sys
main_dir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.insert(1, os.path.join(__file__, main_dir))
sys.path.insert(1, os.path.join(main_dir, 'libs'))
