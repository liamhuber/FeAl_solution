# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base.job.jobtype import JobFactory as JobFactoryCore
from pyiron_feal.utils import HasProject, JobName
import numpy as np

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

    def bcc(self, potl_index=0, a=None, repeat=1):
        job = self.project.create.job.Lammps(self.name.BCC.a(a).repeat(repeat).string)
        job.structure = self.project.create.structure.FeAl.BCC(a=a).repeat(repeat)
        job.potential = self.project.input.potentials[potl_index]
        job.calc_minimize()
        return job
