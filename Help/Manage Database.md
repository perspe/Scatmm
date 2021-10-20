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

When "Add Material" is clicked, a new interface will open (shown bellow) to
provide the necessary information to import the material information.

The procedure to add a material is as follows:

> 1. First choose the file with the material data (should adhere to the format
>   explained in [Importing Data](Import.html)). Only three columns of the file
>   will be considered and they should be on the following order Wavelength/Real
>   Refractive Index/Imaginary Refractive Index.
> 2. Secondly choose a name to identify the material in the database.
> 3. Choose the provided wavelength units
> 4. (Optional) Click preview to check if all the data if begin imported
>   adequately
> 5. Lastly click "Import" to finalize importing the material

![Add Material](add_database.png)

[Return to Home page](help.md)
