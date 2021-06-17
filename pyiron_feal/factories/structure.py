# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.atomistics.structure.factory import StructureFactory as FactoryCore
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


class StructureFactory(FactoryCore):

    def __init__(self):
        super().__init__()
        self._feal_structures = _FeAlStructures(self)

    @property
    def FeAl(self):
        return self._feal_structures


class _FeAlStructures:
    """
    Returns project-relevant FeAl phases.

    Default cell size is 2x2x2 times the BCC unit cell in all cases, since that's the minimal cell for the D0_3 and
    consistency is a beautiful thing.
    """
    _Al_at_frac = 0.18

    def __init__(self, factory: StructureFactory):
        self._factory = factory

    def BCC(self, a=None, repeat=4):
        return self._factory.bulk('Fe', a=a, cubic=True).repeat(repeat)

    def B2(self, a=None, repeat=4):
        struct = self.BCC(a=a, repeat=1)
        struct[1] = 'Al'
        return struct.repeat(repeat)

    def D03(self, a=None, repeat=2):
        struct = self.BCC(a=a, repeat=1).repeat(2)
        manually_identified_Al_sites = [5, 9, 3, 15]
        struct[manually_identified_Al_sites] = 'Al'
        return struct.repeat(repeat)

    def random_BCC(self, a=None, repeat=4):
        struct = self.BCC(a=a, repeat=1).repeat(repeat)
        n_Al = round(self._Al_at_frac * len(struct))
        struct[np.random.choice(range(len(struct)), n_Al, replace=False)] = 'Al'
        return struct

    def FCC(self, a=None, repeat=4):
        return self._factory.bulk(
            'Fe',
            crystalstructure='fcc',
            a=a if a is not None else self._fcc_lattice_constant,
            cubic=True
        ).repeat(repeat)

    @property
    def _fcc_lattice_constant(self):
        d_1NN = self.BCC().get_neighbors(num_neighbors=1, id_list=[0]).distances[0, 0]
        return d_1NN * np.sqrt(2)

    def random_FCC(self, a=None, repeat=4):
        struct = self.FCC(a=a, repeat=1).repeat(repeat)
        n_Al = round(self._Al_at_frac * len(struct))
        struct[np.random.choice(range(len(struct)), n_Al, replace=False)] = 'Al'
        return struct
