The code is written in Python 3.11, and has been tested on Windows 11 and Ubuntu 22.04.

* `loktavolterra_model.py` contains the Lotka-Volterra model.
* `resource_model.py` contains the consumer-resource model.
* `main_simulations.py` contains the functions and code to run the simulation.
* `analysis.py` contains the functions used to analyze the simulation results.
* `figs_simulations.ipynb` provides code to reproduce the figures.

# Installation guide
Python 3 should be installed in the system. Copy the files to a new folder and execute with Python.
If python is installed this process takes only seconds.

# Demo
The python files provide necessary functions for the simulation, and demos are provided in the main execution blocks.
* `figs_simulations.ipynb` contains code blocks to reproduce all figures
* The processed data files in Figshare depository contain daily species composition in all wells for all simulated/predicted invasion scenarios, and are enough for reproducing the figures. However, if raw simulation data is desired, follow the "Instruction for simulation" below.

# Instruction for simulation
* Executing the `main_simulations.py` file would take days on a normal laptop. Reducing the numbers of simulated resident communities (`nrsd`) and invaders (`nivd`) to smaller values like `nrsd == 5, nivs == 2` would allow a small scale simulation for demonstration purpose, which would take around 2 hours.
* Run `main_simulations.py` to obtain raw simulated data files for both Lotka-Volterra and consumer-resource models, including parameters and equilibrium compositions for resident (mod0rsd_.*.pkl) and invader (mod0ivd_.*.pkl),  interaction measurements (mod1comp_.*.pkl), invasion simulation (mod2invasion_.*.pkl), and prediction from interaction measurements (mod3predinv_.*.pkl).
* Run `analysis.py` to calculate the simulated and predicted daily invasion speeds, the parameters should be adjusted according to simulation settings as above, as clarified in the docstring of the `sum_vn` function.
