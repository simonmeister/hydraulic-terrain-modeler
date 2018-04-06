# Hydraulic Terrain Modeler

This (very early, pre-university) work implements a simple 2D grid-based terrain modeler with water simulation and support for water-based erosion, as well as fast online rendering based on GPU tesselation. Terrain can be imported and exported from and to TIF and PNG files. An interactive "brush" enables adding or removing terrain material and sourcing water for on-the-fly modeling, and water sources can be placed for simulation of constant sourcing.

CUDA is used for simulations and visualization are done with modern shader-based OpenGL.
Qt is used for the UI, however this will likely require some patches to make it compile with a modern Qt version.
I will push updates to make it compile as soon as i find time to do this.

A report in **German** is available [here](https://drive.google.com/open?id=11JHhqISbFBbvi_j3kJsj8EUanhBGpt9j).
