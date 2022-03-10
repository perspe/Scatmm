---
title: Manage Database
---

## Accessing the Database

The database can be accessed by the Main interface Database → Manage Database
menu, opening a new window where the database materials can be controled.

Each of the most important information for each database material is shown in 3 different
columns, "Material", "Min Wav", "Max Wav". The first shows the material name,
and the next two provide the wavelength range where the stored database values
are defined. These last values are particularly useful to guarantee that the
material is adequately defined for a possible simulation range. Simulations and
optimizations are limited to the wavelength range where all materials in that
particular simulation/optimization are defined.

## Database Actions

The database provides three different interactions:

> * __Add Material__: To add a new material to the database (see more below)\
> * __Remove Material__: Remove the currently selected material from the database\
> * __View Material__: Preview the stored real refractive index, complex\
>  refractive index and their respective interpolations for a particular\
>  material\

![Manage Database](manage_database.png)

## Adding New Materials

When "Add Material" is clicked, a new interface will open to
provide the necessary information to import the material information.

More information is provided in [Importing Data](./Import.md)

[Return to Home page](help.md)
