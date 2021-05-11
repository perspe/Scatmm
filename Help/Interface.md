---
title: Interface
---

# Main Interface

When the program is opened it will show the following interface

![Basic GUI](basic_gui.png)

On the topmost part of the interface there's the File/Database/Help menus.
The File menu has the properties menu, where some of the basic simulation/optimization parameters can be changed from their default values
 and the Database menu can be used to manipulate the database materials (either to add/preview/remove materials)
 
# Simulation Properties

On the right there's the Simulation properties (image bellow). The different simulation properties are the minimum and maximum simulation wavelength (that essentially define the simulation range), the incident angles (theta and phi) and 2 light polarization components (TE and TM). These values correspond to the basic simulation conditions for the setup. Lastly, on the bottommost part of the simulation properties are the different possible results that can be obtained (Reflection, Transmission and Absorption). For a simulation, checking one of these parameters indicates to the program that the respective result should be showed when doing a simulation (more on [Simulation](Simulation.html))

![Simulation Properties](simulation_properties.png)


The program can be used either to do fast simulations of 1D layered devices or to try and determine the thickness of different layers from input data. Each possible use can be chosen by selecting the respective tab (simulation/optimization) in the interface.

![Use Tabs](sim_opt_tabs.png)

# Simulation Region

The simulation is defined by 3 main regions. Firstly, the reflection region (the topmost region) that defines the medium above the device and from where light comes. Secondly the transmission region that defines the medium after light passes through the device. Both the media are defined in terms of their non-dispersive real and complex refractive indexes (the default values are for an air medium). Lastly, in the middle the device can be defined (by default a 2-layer material is defined as seen in the image bellow). Layers can be added/removed to/from the bottom of the stack by clicking the respective buttons (Add Layer and Remove Layer). Each layer is composed of 2 parts. On the left the material for the layer can be chosen from the available materials stored in the database ([Database](Manage Database.html)), and on the right the thickness of the layer can be chosen.

![Layers](mat_setup.png)

On the bottom right of the interface there is a import button that can be used to import data from a file ([Import](Import.html)). This data can be used to compare external results/measurements with those of the simulation, or they can be used as fitting measure for the optimization.

[Return to Home page](help.html)
