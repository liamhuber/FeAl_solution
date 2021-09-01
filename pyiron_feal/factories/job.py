# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base.job.jobtype import JobFactory as JobFactoryCore
from pyiron_feal.utils import HasProject, JobName

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Jun 23, 2021"


class JobFactory(JobFactoryCore):
    def __init__(self, project):
        super().__init__(project)
        self._minimize = _Minimize(project)
        self._mcmd = _MCMD(project)

    @property
    def minimize(self):
        return self._minimize

    @property
    def mcmd(self):
        return self._mcmd


class _Minimize(HasProject):
    """
    Not strictly job creation routines -- if the job already exists its HDF path is returned instead.
    """

    def __init__(self, project):
        super().__init__(project)
        self.name = JobName('min')

    def _lammps_minimization(self, potl_index, name, structure, pressure, delete_existing_job=False):
        hdf_path = self.project.inspect(name)
        if hdf_path is None or delete_existing_job:
            job = self.project.create.job.Lammps(name, delete_existing_job=delete_existing_job)
            job.structure = structure
            job.potential = self.project.input.potentials[potl_index]
            job.calc_minimize(pressure=pressure, n_print=1e5)
            return job
        else:
            return hdf_path

    def bcc(self, potl_index=0, a=None, repeat=1, trial=None, pressure=0, delete_existing_job=False, c_Al=None):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                bcc=True,
                a=a,
                repeat=repeat,
                trial=trial,
                pressure=pressure,
                c_Al=c_Al
            ),
            structure=self.project.create.structure.FeAl.bcc(a=a, repeat=repeat, c_Al=c_Al),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def d03(
            self,
            potl_index=0,
            a=None,
            repeat=1,
            trial=None,
            pressure=0,
            delete_existing_job=False,
            c_D03_anti_Al_to_Fe=None,
            c_D03_anti_aFe_to_Al=None,
            c_D03_anti_bFe_to_Al=None
    ):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                d03=True,
                a=a,
                repeat=repeat,
                trial=trial,
                pressure=pressure,
                c_D03_anti_Al_to_Fe=c_D03_anti_Al_to_Fe,
                c_D03_anti_aFe_to_Al=c_D03_anti_aFe_to_Al,
                c_D03_anti_bFe_to_Al=c_D03_anti_bFe_to_Al
            ),
            structure=self.project.create.structure.FeAl.d03(
                a=a,
                repeat=repeat,
                c_D03_anti_Al_to_Fe=c_D03_anti_Al_to_Fe,
                c_D03_anti_aFe_to_Al=c_D03_anti_aFe_to_Al,
                c_D03_anti_bFe_to_Al=c_D03_anti_bFe_to_Al
            ),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def b2(
            self,
            potl_index=0,
            a=None,
            repeat=1,
            trial=None,
            pressure=0,
            delete_existing_job=False,
            c_B2_anti_Al_to_Fe=None,
            c_B2_anti_Fe_to_Al=None,
    ):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                b2=True,
                a=a,
                repeat=repeat,
                trial=trial,
                pressure=pressure,
                c_B2_anti_Al_to_Fe=c_B2_anti_Al_to_Fe,
                c_B2_anti_Fe_to_Al=c_B2_anti_Fe_to_Al,
            ),
            structure=self.project.create.structure.FeAl.b2(
                a=a, repeat=repeat, c_B2_anti_Al_to_Fe=c_B2_anti_Al_to_Fe, c_B2_anti_Fe_to_Al=c_B2_anti_Fe_to_Al
            ),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def fcc(self, potl_index=0, a=None, repeat=1, trial=None, pressure=0, delete_existing_job=False, c_Al=None):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                fcc=True,
                a=a,
                repeat=repeat,
                trial=trial,
                pressure=pressure,
                c_Al=c_Al
            ),
            structure=self.project.create.structure.FeAl.fcc(a=a, repeat=repeat, c_Al=c_Al),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def columnar_b2(self, potl_index=0, pressure=0, delete_existing_job=False, planar_repeats=1):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                columnar=True,
                repeat=planar_repeats,
                pressure=pressure,
            ),
            structure=self.project.create.structure.FeAl.columnar_b2(planar_repeats=planar_repeats),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def layered(self, potl_index=0, pressure=0, delete_existing_job=False, layers=1):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                layered=True,
                repeat=layers,
                pressure=pressure,
            ),
            structure=self.project.create.structure.FeAl.layered_Al(layers=layers),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def interactive_cluster(
            self,
            potl_index=0,
            a=None,
            repeat=1,
            trial=None,
            pressure=0,
            delete_existing_job=False,
            c_Al=None,
            max_cluster_fraction=0.125,
            symbol_ref=None
    ):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name(
                interactive=True,
                potl_index=potl_index,
                bcc=True,
                a=a,
                repeat=repeat,
                trial=trial,
                pressure=pressure,
                c_Al=c_Al,
                max_cluster_fraction=max_cluster_fraction,
                symbol_ref=symbol_ref
            ),
            structure=self.project.create.structure.FeAl.bcc(a=a, repeat=repeat, c_Al=c_Al),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )


class _MCMD(HasProject):
    """
    Actually job creation routines -- loads the job if it already exists.
    """

    def __init__(self, project):
        super().__init__(project)
        self.name = JobName('mcmd')

    def _lammps_vcsgc(
            self,
            potl_index,
            name,
            structure,
            temperature,
            dmu,
            c_Al,
            delete_existing_job=False,
            n_ionic_steps=1e4,
            n_print=None,
            kappa=1e4,
            temperature_mc=None,
            **other_vcsgc_kwargs
    ):
        n_print = n_ionic_steps if n_print is None else n_print
        temperature_mc = temperature if temperature_mc is None else temperature_mc
        job = self.project.load(name)
        if job is None or delete_existing_job:
            job = self.project.create.job.Lammps(name, delete_existing_job=delete_existing_job)
            job.structure = structure
            job.potential = self.project.input.potentials[potl_index]
            job.calc_vcsgc(
                temperature=temperature,
                mu={'Fe': 0, 'Al': dmu},
                target_concentration={'Fe': 1 - c_Al, 'Al': c_Al},
                n_ionic_steps=n_ionic_steps,
                n_print=n_print,
                kappa=kappa,
                temperature_mc=temperature_mc,
                langevin=True,
                **other_vcsgc_kwargs
            )
        return job

    def small_cube(
            self,
            temperature,
            potl_index=0,
            dmu=0,
            c_Al=0.18,
            delete_existing_job=False,
            n_ionic_steps=1e4,
            n_print=None,
            kappa=1e4,
            temperature_mc=None,
            **other_vcsgc_kwargs
    ):
        """Approx 4x4x4 nm. dmu < 0 drives to Al, dmu > 0 drives to Fe."""
        return self._lammps_vcsgc(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                cube=True,
                temperature=temperature,
                dmu=dmu,
                c_Al=c_Al,
                n_steps=n_ionic_steps,
                kappa=kappa,
                temperature_mc=temperature_mc
            ),
            structure=self.project.create.structure.FeAl.bcc(repeat=7, c_Al=c_Al),
            temperature=temperature,
            dmu=dmu,
            c_Al=c_Al,
            delete_existing_job=delete_existing_job,
            n_ionic_steps=n_ionic_steps,
            n_print=n_print,
            kappa=kappa,
            temperature_mc=temperature_mc,
            **other_vcsgc_kwargs
        )

    def experimental_size(
            self,
            temperature,
            potl_index=0,
            dmu=0,
            c_Al=0.18,
            delete_existing_job=False,
            n_ionic_steps=1e4,
            n_print=None,
            kappa=1e4,
            temperature_mc=None,
            **other_vcsgc_kwargs
    ):
        """Approx 50x10x10 nm. dmu < 0 drives to Al, dmu > 0 drives to Fe."""
        return self._lammps_vcsgc(
            potl_index=potl_index,
            name=self.name(
                potl_index=potl_index,
                experimental=True,
                temperature=temperature,
                dmu=dmu,
                c_Al=c_Al,
                n_steps=n_ionic_steps,
                kappa=kappa,
                temperature_mc=temperature_mc
            ),
            structure=self.project.create.structure.FeAl.bcc(repeat=(87, 18, 18), c_Al=c_Al),
            temperature=temperature,
            dmu=dmu,
            c_Al=c_Al,
            delete_existing_job=delete_existing_job,
            n_ionic_steps=n_ionic_steps,
            n_print=n_print,
            kappa=kappa,
            temperature_mc=temperature_mc
            **other_vcsgc_kwargs
        )