# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics import Project as ProjectCore
from pyiron_feal.factories.structure import StructureFactory
from pyiron_feal.factories.job import JobFactory
from pyiron_base import DataContainer
from pyiron_feal.subroutines import ZeroK
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
__date__ = "Jun 10, 2021"


class ProjectInput(DataContainer):
    def __init__(self, init=None, table_name=None):
        super().__init__(init=init, table_name=table_name)
        self.potentials_eam = np.array([
            '2005--Mendelev-M-I--Al-Fe--LAMMPS--ipr1',
            '2020--Farkas-D--Fe-Ni-Cr-Co-Al--LAMMPS--ipr1',
        ])
        self.potentials_meam = np.array([
            '2010--Lee-E--Fe-Al--LAMMPS--ipr1',
            '2012--Jelinek-B--Al-Si-Mg-Cu-Fe--LAMMPS--ipr2',
        ])
        self.experimental_data = {
            'c_Al': 0.18,
            'T': 523,
            'SS': 1 - (0.0042 + 0.1232),
            'B2': 0.0042,
            'D03': 0.1232
        }

    @property
    def potentials(self):
        return np.append(self.potentials_eam, self.potentials_meam)


class Project(ProjectCore):

    def __init__(self, path="", user=None, sql_query=None, default_working_directory=False):
        super(Project, self).__init__(
            path=path,
            user=user,
            sql_query=sql_query,
            default_working_directory=default_working_directory
        )
        self.create._structure = StructureFactory()
        self.create._job_factory = JobFactory(self)
        self._zerok = ZeroK(self)

    @property
    def input(self) -> ProjectInput:
        # A workaround since we can't populate the data field in `__init__`
        try:
            return self.data.input
        except AttributeError:
            self.data.input = ProjectInput()
            return self.data.input

    @property
    def zerok(self):
        return self._zerok
