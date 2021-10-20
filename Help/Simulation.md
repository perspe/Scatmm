---
title: Simulation
---

## Starting a Simulation

The program allows for 2 different modes of simulations.
*Wavelngth* and *Angle* Simulations. Most parameters are shared
between modes, with only a few changing with the mode. As such, 
a simulation is defined by the following parameters:

> - min/max wavelength (for *Wavelength* simulations)
> - single wavelength (for *Angle* simulations)
> - incidence angle (phi/theta) (theta is only available in *Wavelength* simulations)
> - incident light polarization (TE/TM)
> - layer setup (reflection medium/device/transmission medium)
> - measurables (reflection/transmission/reflection)

## Simulation

When the "simulate" button is clicked the program will gather the
aforementioned information, use it to perform the simulation and then plot
the results. The plot will show the results for the different measurables
chosen, for instance is reflection/transmission/absorption are all checked then
the simulation will plot the results for all three values, if only reflection is
chosen then only the reflection will be plotted...

## Multiple Simulations

Multiple simulations can be done sequentially and the results will be added to the *Results Preview* region
and stored internally. The number of simulations stored are
shown in the clear button ("clear(10)" means there are 10 simulations stored in
memory). Clicking the clear button will flush all the results and clear all the
open plots.

![Multiple Simulations](stored_sims.png){width=100%}

## Exporting Data

The export button can be used to export simulation results to files. When
clicked it will open the bellow shown interface.

On the left it is possible to choose with measurables to export (reflection/transmission/absorption) and if
the absorption for each layer should also be exported along with the simulated data. On the right
all the simulations are shown with a shortcut name briefly summarizing the simulation setup

> ($A/W$) (*$SN$*) (theta, phi) |material name(material thickness)|...\
>
> * $A/W$ indicates if the simulation was Angle or Wavelength, respectivelly
> * $SN$ indicates the number of the simulation

Upon selecting a particular simulation, the tex box bellow will be filled with all the simulation details,
providing a full report of the simulation conditions. Having chosen a specific simulation, the
absorption/reflection/transmission profiles can be previewed (*Preview* button) or exported (*Export* button).
A *Export All* button is also provided as a shortcut to export all the stored simulations.

![Export](export.png)

[Return to Home page](help.html)
