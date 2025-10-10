Collection of kernels that are designed to poke different parts of the GPU
architecture to see how they react. Draft-y, but easy to figure out:

- Kernels are run from `main.cu` and defined in `architecture-boundaries.cuh`
- Plotting script is pretty adhoc, but maybe I'll improve it later for faster
iteration times + easier to pick up by others
- Run script won't work with current plotting utility, but it shows the basics
of what I'm using to compile the kernels (plus or minus some things) 

I'll add some nice plots for each kernel I've written when I have some time to
improve the plotting script.
