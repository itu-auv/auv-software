# AUV Command Line Tools

Bash command line tools for controlling the auv.


`note:` These command line tools, requires ITU AUV Team Software 
Stack installed and running in order for to work.

## Install
```
python3 setup.py install --user
```

## Usage
```sh
usage: auv <command> [<args>]

positional arguments:
  {arm,disarm,set_depth,set_armed,get_armed,get_depth}
                        sub-command help
    arm                 Arms vehicle
    disarm              Disarms vehicle
    set_depth           set_depth command help
    set_armed           set_armed command help
    get_armed           returns current arming status
    get_depth           returns current depth status

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose output
  -t TIMEOUT, --timeout TIMEOUT
                        sets Message/Service connection timeout for <TIMEOUT>
                        seconds.
```