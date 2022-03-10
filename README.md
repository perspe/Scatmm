# Transfer Matrix Method Graphical Interface

## Usage Example

![Usage Demonstration](Help/preview.gif)

## Description

This script implements a graphical interface (on PyQt5) to interact with the transfer matrix method (calculate Reflection/Transmission/Absorption for 1D stacked material layers). The transfer matrix method was developed from the information provided on the Computational Electromagnetics Youtube video series by [EMPossible](https://www.youtube.com/channel/UCPC6uCfBVSK71MnPPcp8AGA).
The program has a database of materials (easily expanded with new materials), that is used to perform the calculations.
This program allows for:
* Calculation of Reflection/Transmission/Absorption for arbitrarily sized stacks of planar materials
* Calculation of Absoprtion for a particular layer in the stack
* Calculation of Reflection/Transmission/Absorption for broad-angle/broadband simulations
* Export the results of simulations to text files
* Import data to compare with simulations
* Thickness fitting from imported data (using a particle swarm algorithm to find the thickness combination for all layers that minimizes the error between the simulation algorithm and the imported data.
* It also has a interface for the python structures used in the calculations (Described API_Tutorial.html and API_Tutorial.ipynb files)

The program was written using the following version of different python packages (although it should also work with any recent version of these packages)

* python 3.8.12
* pandas 1.3.5
* numpy 1.21.4 (minimal version of 1.20 needed for typing module)
* scipy 1.7.3
* matplotlib 3.5
* pyqt 5.9.2
* appdirs 1.4.4

__Note:__ The program also has a module built under cython. There are compiled binaries for Windows 10, and Linux (used in Mint and Arch). In any case, it should be possible to compile it (provided a compiler is installed, like g++) by running ```python setup.py``` in the modules folder.

The program can either be ran from the terminal by running the scatmm.py script

```python
python scatmm.py
```

The script can be also run from IDEs, although it is important to guarantee that all the required packages are properly installed in the virtual environment.

It also possible to run everything from the backend functions (as exmplained in API_Tutorial.html and API_Tutorial.ipynb)

When run the program will display the image bellow.
The help menu provides a brief description of all the program's capabilities

![Scatmm Interface](./Help/basic_)
