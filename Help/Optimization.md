---
title: Optimization
---

## Defining the Optimization

The optimization is only available for *Wavelength* simulations.
To define a optimization, the following parameters are required

> - incidence angle (phi/theta)\
> - incident light polarization (te/tm)\
> - layer setup (reflection medium/device/transmission medium)\
> - optimization measurable (one of reflection/transmission/reflection)\
> - data to fit\

## Steps to Start an Optimization

Before doing an optimization data should be imported to perform the fitting ([Import Data](Import.md)).
The imported data should have 2 columns, one with the wavelength in nm and another one
with the measurement to be fitted (absorption/reflection/transmission). It is important
to note that any extra columns in the file will be ignored.
Afterwards the user should select which measurable is to be fitted, from the simulation properties.
For instance, if the imported data provides some reflection data, then
the user should only select reflection in the simulation properties to fit reflection results.

Afterwards, the layer setup should be built, by selecting the materials for the
several layers and selecting the minimum and maximum thickness values for each
layer. It is important to note that the program will never find values outside
this range.

## Optimization

The optimization is started by clicking the "Optimize" button. After doing
that, the progress bar will provide information regarding the progress of the
optimization.

When the optimization finishes it will plot the best result achieved alongside
the imported data and will provide the best found thicknesses in the text box
on the bottom right. It is important to note that the results will not be added
to the simulation stack. If the results are required, the provided optimized
thicknesses, provided in the text box above the *Import* button can be used to make
a new simulation.

[Return to Home page](help.html)

