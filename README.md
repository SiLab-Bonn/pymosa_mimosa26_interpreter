
# pyBAR_mimosa26_interpreter [![Build Status](https://travis-ci.org/SiLab-Bonn/pyBAR_mimosa26_interpreter.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/pyBAR_mimosa26_interpreter) [![Build Status](https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/pyBAR_mimosa26_interpreter?svg=true)](https://ci.appveyor.com/project/DavidLP/pyBAR_mimosa26_interpreter-71xwl)

pyBAR_mimosa26_interpreter - A Mimosa26 raw data interpreter in Python.

This package makes it possible to interpret Mimosa26 raw data (from a Mimosa26 telescope) recorded with `pymosa <https://github.com/SiLab-Bonn/pymosa>`_. 
Additionally, an event building is done using trigger words from the TLU. Moreover, it contains the possibilty to create for each Mimosa26 plane an occupancy map.

**Note:**
For the interpretation it is needed that the trigger data format is set to 2 (15 bit trigger number + 16 bit trigger timestamp)

## Installation

Install the required packages:
  ```
  conda install numba numpy tables matplotlib tqdm
  ```

Then install the Mimosa26 interpreter:
  ```
  python setup.py develop
  ```

## Usage

An example script which does the raw data interpretation as well as the creation of a hit table which can be used for `testbeam analysis <https://github.com/SiLab-Bonn/testbeam_analysis>`_
is located in the `example folder <https://github.com/SiLab-Bonn/pyBAR_mimosa26_interpreter/blob/new_mimosa26_interpreter/examples/example.py>`_


## Documentation

Documentation can be found under:
