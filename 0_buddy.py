# Python standard library
from sys import argv, stdout, stderr
from time import sleep
from subprocess import Popen
from pathlib import Path

options = [x[2:] for x in argv[1:] if x.startswith('--')]
plain   = [x for x in argv[1:] if not x.startswith('--')]

clicky_path = Path('1_mister_clicky.py')
forest_path = Path('2_random_forest_clickbooster.py')

assert clicky_path.exists(), 'Cannot find napari interface, %s'%clicky_path
assert forest_path.exists(), 'Cannot find random forester, %s'%forest_path

def run(target, args):
    cmd = ['python', target] + plain
    return Popen(cmd, stdout = stdout, stderr = stderr)

clicky = run(clicky_path, plain)

sleep(2.5)

forest = run(forest_path, plain)

while clicky.poll() == None:
    sleep(0.5)
    
if forest.poll() == None:
    forest.terminate()
