---
title: Optimization
---

# Defining the Optimization

An optimization is defined by the following parameters:

- incidence angle (phi/theta)
- incident light polarization (te/tm)
- layer setup (reflection medium/device/transmission medium)
- optimization measurable (one of reflection/transmission/reflection)
- data to fit

# Steps to Start an Optimization

Before doing an optimization data should be imported to perform the fitting. The data should have 2 columns, one with the wavelength in nm and another one with the measurement to be fitted (absorption/reflection/transmission). Afterwards the user should select, from the simulation properties which is the measurable to be fitted. If the imported data, provides some reflection data, the user should only select reflection in the simulation properties.

Afterwards, the layer setup should be built, by selecting the materials for the several layers and selecting the minimum and maximum thickness values for each layer. It is important to note that the program will never find values outside this range.

# Optimization

The optimization is started by clicking the "Optimize" button. After doing that, the progress bar will provide information regarding the progress of the optimization.

When the optimization finishes it will plot the best result achieved alongside the imported data and will provide the best found thicknesses in the text box on the bottom right.

[Return to Home page](help.html)

