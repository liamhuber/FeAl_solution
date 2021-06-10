# FeAl solution study

In collaboration with Bat et al. we will look at FeAl solid solution along with B2 and D03 phases.
The current working plan is:

- ~~Liam (or Raynol)) Makes a quick notebook of 0K calculations for the relevant phases and passes it to Raynol~~
- Raynol) MD for volume distributions at relevant temperatures; TILD for bulk Fe total free energy; TILD for chemical change to get B2 or D03
  - This might need some conceptual adjustment if the B2/D03 volumes are way off from bulk Fe. As long as the distributions overlap, we can run TILD at the point of maximum overlap in the distributions and apply the C&C conversion from Helmholtz to Gibbs
- Raynol and Liam) Analysis of phase stabilities in $(T, \Delta\mu_{Al})$-space to nail down the most useful chemical potentials for MC/MD
  - With the variance constrained MC/MD the chemical potential is not absolutely critical, but (a) this analysis will already tell us if we have even a chance of the phases both forming, and (b) having a good chempot still speeds up the MC/MD
- Marvin (or Liam)) foreach potential foreach temperature MC/MD calculations
- Marvin (or Liam)) Chemically-resolved structural analysis to look for signatures of the secondary phases
  - Then we pass off the point cloud data to the experimentalists, maybe with some sort of slicing into a needle-shape so they don't have to learn about PBCs

# pyiron publication template

This template is forked from the official pyiron template and adapted for my own research projects.
In particular, it is assumed that there is custom `.py` code that should have unit tests.

## Step by step

* ~~Create a new project with this template.~~
* ~~Update the `environment.yml` to the basic resources you will need.~~
* ~~Refactor --> Rename `pyiron_src` to something sensible for your project.~~
* Update `example.ipynb` to run something simple using your new pyiron library.
* Develop and science your heart out.
    * Everything in the `projects` folder is gitignored, so put your projects there 
    * The notebook `scratch.ipynb` is also ignored
    * Be mindful about the volume of data (e.g. calculation results) you push and pull.
* Document your process using LaTeX in the `writeup` directory
    * Update the `boilerplate` and `authors` to suite your needs.
      * Note that the title is stored in the boiler plate!
    * Consider using the `supplementary` directory to track your work as you do it, so you don't need to go back and figure things out.
    * TODO: Use `pylatex` to automatically scrape data to minimize human error.

## mybinder

Is not yet working for me at all.
I left these parts as-is from the orginal template.

## jupyterbook

Worked when I went to the GitHub settings and created the book there.
Nonetheless, I haven't changed anything here relative to the official tempalte.

## Everything else

E.g. how to import existing data?
Go look at the README for the official template.

## License
pyiron, the pyiron publication template, and this fork of it are all licensed under the BSD-3-Clause license which is included in the `LICENSE` file. 
In addition an `CODE_OF_CONDUCT.md` file is included to foster an open and welcoming environment.
