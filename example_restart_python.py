import os
import subprocess
import sys
import eden

def main():
    env = os.environ.copy()

    if 'PYTHONHASHSEED' not in env:
        env['PYTHONHASHSEED'] = "0"
        proc = subprocess.Popen([sys.executable,sys.argv[0]],
                                env=env)
        proc.communicate()
        exit(proc.returncode)

    # check if hash has been properly de-randomized in python 3
    h = hash(eden.__magic__)
    assert h == eden.__magic_py2hash__ or h == eden.__magic_py3hash__, 'Unexpected hash value: "{}". Please check if python 3 hash normalization is disabled by setting shell variable PYTHONHASHSEED=0.'.format(h)
    exit(0)

if __name__ == "__main__":
    main()
