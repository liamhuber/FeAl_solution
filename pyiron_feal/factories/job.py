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

    @property
    def minimize(self):
        return self._minimize


class _Minimize(HasProject):
    def __init__(self, project):
        super().__init__(project)
        self.name = JobName('min')

    def _lammps_minimization(self, potl_index, name, structure, pressure, delete_existing_job=False):
        job = self.project.create.job.Lammps(name, delete_existing_job=delete_existing_job)
        job.structure = structure
        job.potential = self.project.input.potentials[potl_index]
        job.calc_minimize(pressure=pressure)
        return job

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

    def BCC(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).BCC.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.BCC(a=a).repeat(repeat),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def FCC(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).FCC.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.FCC(a=a).repeat(repeat),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def random_BCC(
            self, potl_index=0, a=None, repeat=2, c_Al=None, pressure=0, trial=None, delete_existing_job=False
    ):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).random_BCC.a(a).repeat(repeat).c_Al(c_Al).trial(trial).string,
            structure=self.project.create.structure.FeAl.random_BCC(a=a, repeat=repeat, c_Al=c_Al),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def B2(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).B2.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.B2(a=a).repeat(repeat),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def D03(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.D03(a=a).repeat(repeat),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def dilute(self, potl_index=0, a=None, repeat=2, pressure=0, delete_existing_job=False):
        structure = self.project.create.structure.FeAl.BCC(a=a).repeat(repeat)
        structure[0] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).BCC.append('Al').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def B2_anti_Fe_to_Al(self, potl_index=0, a=None, repeat=2, pressure=0, delete_existing_job=False):
        structure = self.project.create.structure.FeAl.B2(a=a).repeat(repeat)
        structure[0] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).B2.append('Al').a(a).repeat(repeat).P(pressure).string,
            structure=structure,
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def B2_anti_Al_to_Fe(self, potl_index=0, a=None, repeat=2, pressure=0, delete_existing_job=False):
        structure = self.project.create.structure.FeAl.B2(a=a).repeat(repeat)
        structure[1] = 'Fe'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).B2.append('Fe').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def D03_anti_aFe_to_Al(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        structure = self.project.create.structure.FeAl.D03(a=a).repeat(repeat)
        structure[0] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.append('aAl').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    # The two unique Fe sites were found in a by-hand analysis of the chemistry of the first two neighbour shells and
    # then a brute-force calculation of the energy which simply confirmed the observed symmetries.
    def D03_anti_bFe_to_Al(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        structure = self.project.create.structure.FeAl.D03(a=a).repeat(repeat)
        structure[1] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.append('bAl').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def D03_anti_Al_to_Fe(self, potl_index=0, a=None, repeat=1, pressure=0, delete_existing_job=False):
        structure = self.project.create.structure.FeAl.D03(a=a).repeat(repeat)
        structure[3] = 'Fe'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.append('Fe').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )

    def random_D03_antisites_Al_to_Fe(
            self, potl_index=0, a=None, repeat=2, c_antisites=None, pressure=0, trial=None, delete_existing_job=False
    ):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.a(a).repeat(repeat).c_D03_anti_Al_to_Fe(c_antisites).trial(trial).string,
            structure=self.project.create.structure.FeAl.random_D03_antisites_Al_to_Fe(
                a=a, repeat=repeat, c_antisites=c_antisites
            ),
            pressure=pressure,
            delete_existing_job=delete_existing_job
        )
