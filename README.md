# FeAl solution study

This repository stores all the code necessary to reproduce (at least stochastically) all the data, figures, and text for the atomistic simulations associated with **MANSUCRIPTTBA**.

Steps to run:

0. Be in a unix environment
1. Clone repo
2. Create the necessary python environment using `environment.yml`
3. (Optional, if you really want to re-run the whole project from scratch) If you have an existing pyiron installation on a compute cluster, and already have the queue adapter configured, then set the `SERVER` value in `project.ipynb` to match the server queue appropriate for single-core jobs taking <4 days. Otherwise set this value to `None`/remove all instances of `server.queue = ` from the notebook, and make sure your machine is ready for a week of up-time...

With steps (0-2) you can already play around in `example.ipynb`, which will recreate one of the early manuscript figures with minimal computational cost.

The purpose here is simply to provide scientific reproducibility, not to supply a fully extensible/reusable/magical and beautiful software package, but nonetheless at least a little effort has been put into documentation and usability -- we hope it's helpful.

## License
pyiron, the pyiron publication template, and this fork of it are all licensed under the BSD-3-Clause license which is included in the `LICENSE` file. 
In addition an `CODE_OF_CONDUCT.md` file is included to foster an open and welcoming environment.
