---
title: Importing Data
---

# How Data Should be Formated

Data should be imported from text files. The import functionality has a limited
range of applications, thus leaving the burden to the user to adequately
prepare the data for importing. The data should have a tabular format of
numeric-only entries separated either by spaces or commas. Any non-numeric data
should be commented with # (see Example 3 bellow). Furthermore, when importing
the data the program does not check the number of columns in the file. For the
case of optimizations, only the first two columns will be used and will be
assumed to be wavelength (in nm) and either absorption/reflection/transmission
([Optimization](Optimization.html)). To import materials to the database 3
columns will be used and assumed to be wavelength/real refractive index/complex
refractive index ([Manage Database](Manage Database.html)). Below are provided
3 examples of possible data to be imported.

# Examples

- Example 1:

251.57 2.31717213 0.577801613

253.16 2.33168536 0.569701795

254.75 2.34626565 0.560078808

256.34 2.36067795 0.548891345

257.93 2.37467908 0.536140733

259.52 2.38802531 0.521875087

261.12 2.40048438 0.506187696

262.71 2.41184546 0.489213773

264.3 2.42192779 0.471124947

265.89 2.43058781 0.45212117


- Example 2:

251.57,2.31717213,0.577801613

253.16,2.33168536,0.569701795

254.75,2.34626565,0.560078808

256.34,2.36067795,0.548891345

257.93,2.37467908,0.536140733

259.52,2.38802531,0.521875087

261.12,2.40048438,0.506187696

262.71,2.41184546,0.489213773

264.3,2.42192779,0.471124947

265.89,2.43058781,0.45212117


- Example 3:

#Column1 Column2 Column3

251.57 2.31717213 0.577801613

253.16 2.33168536 0.569701795

254.75 2.34626565 0.560078808

256.34 2.36067795 0.548891345

257.93 2.37467908 0.536140733

259.52 2.38802531 0.521875087

261.12 2.40048438 0.506187696

262.71 2.41184546 0.489213773

264.3 2.42192779 0.471124947

265.89 2.43058781 0.45212117

[Return to Home page](help.html)
