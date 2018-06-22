import eden
import subprocess
import sys
import os


def main():
    env = os.environ.copy()

    # in case the PYTHONHASHSEED was not set, set to 0 to denote
    # that hash randomization should be disabled and
    # restart python for the changes to take effect
    if 'PYTHONHASHSEED' not in env:
        env['PYTHONHASHSEED'] = "0"
        proc = subprocess.Popen([sys.executable] + sys.argv,
                                env=env)
        proc.communicate()
        exit(proc.returncode)

    # check if hash has been properly de-randomized in python 3
    # by comparing hash of magic tuple
    h = hash(eden.__magic__)
    assert h == eden.__magic_py2hash__ or h == eden.__magic_py3hash__, 'Unexpected hash value: "{}". Please check if python 3 hash normalization is disabled by setting shell variable PYTHONHASHSEED=0.'.format(h)

    # run program and exit
    print("This is the magic python hash restart script.")
    exit(0)


if __name__ == "__main__":
    main()
