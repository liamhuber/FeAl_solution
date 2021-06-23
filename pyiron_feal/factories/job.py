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

    def _lammps_minimization(self, potl_index, name, structure, pressure):
        job = self.project.create.job.Lammps(name)
        job.structure = structure
        job.potential = self.project.input.potentials[potl_index]
        job.calc_minimize(pressure=pressure)
        return job

    def BCC(self, potl_index=0, a=None, repeat=1, pressure=0):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).BCC.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.BCC(a=a).repeat(repeat),
            pressure=pressure
        )

    def FCC(self, potl_index=0, a=None, repeat=1, pressure=0):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).FCC.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.FCC(a=a).repeat(repeat),
            pressure=pressure
        )

    def random_BCC(self, potl_index=0, a=None, repeat=2, Al_at_frac=None, pressure=0):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).random_BCC.a(a).repeat(repeat).c_Al(Al_at_frac).string,
            structure=self.project.create.structure.FeAl.random_BCC(a=a, repeat=repeat, Al_at_frac=Al_at_frac),
            pressure=pressure
        )

    def B2(self, potl_index=0, a=None, repeat=1, pressure=0):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).B2.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.B2(a=a).repeat(repeat),
            pressure=pressure
        )

    def D03(self, potl_index=0, a=None, repeat=1, pressure=0):
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.a(a).repeat(repeat).string,
            structure=self.project.create.structure.FeAl.D03(a=a).repeat(repeat),
            pressure=pressure
        )

    def dilute(self, potl_index=0, a=None, repeat=2, pressure=0):
        structure = self.project.create.structure.FeAl.BCC(a=a).repeat(repeat)
        structure[0] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).BCC.append('Al').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure
        )

    def B2_anti_Fe_to_Al(self, potl_index=0, a=None, repeat=2, pressure=0):
        structure = self.project.create.structure.FeAl.B2(a=a).repeat(repeat)
        structure[0] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).B2.append('Al').a(a).repeat(repeat).P(pressure).string,
            structure=structure,
            pressure=pressure
        )

    def B2_anti_Al_to_Fe(self, potl_index=0, a=None, repeat=2, pressure=0):
        structure = self.project.create.structure.FeAl.B2(a=a).repeat(repeat)
        structure[1] = 'Fe'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).B2.append('Fe').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure
        )

    def D03_anti_aFe_to_Al(self, potl_index=0, a=None, repeat=1, pressure=0):
        structure = self.project.create.structure.FeAl.D03(a=a).repeat(repeat)
        structure[0] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.append('aAl').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure
        )

    # The two unique Fe sites were found in a by-hand analysis of the chemistry of the first two neighbour shells and
    # then a brute-force calculation of the energy which simply confirmed the observed symmetries.
    def D03_anti_bFe_to_Al(self, potl_index=0, a=None, repeat=1, pressure=0):
        structure = self.project.create.structure.FeAl.D03(a=a).repeat(repeat)
        structure[1] = 'Al'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.append('bAl').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure
        )

    def D03_anti_Al_to_Fe(self, potl_index=0, a=None, repeat=1, pressure=0):
        structure = self.project.create.structure.FeAl.D03(a=a).repeat(repeat)
        structure[3] = 'Fe'
        return self._lammps_minimization(
            potl_index=potl_index,
            name=self.name.potl(potl_index).D03.append('Fe').a(a).repeat(repeat).string,
            structure=structure,
            pressure=pressure
        )
