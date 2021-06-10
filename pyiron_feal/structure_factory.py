# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.atomistics.structure.factory import StructureFactory as FactoryCore

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

    def BCC(self, a=None):
        return self._factory.bulk('Fe', a=a, cubic=True).repeat(2)

    def B2(self, a=None):
        struct = self._factory.bulk('Fe', a=a, cubic=True)
        struct[1] = 'Al'
        return struct.repeat(2)

    def D03(self, a=None):
        struct = self.BCC(a=a)
        manually_identified_Al_sites = [5, 9, 3, 15]
        struct[manually_identified_Al_sites] = 'Al'
        return struct
