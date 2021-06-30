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
    _c_Al = 0.18

    def __init__(self, factory: StructureFactory):
        self._factory = factory

    def BCC(self, a=None):
        return self._factory.bulk('Fe', a=a, cubic=True)

    def B2(self, a=None):
        struct = self.BCC(a=a)
        struct[1] = 'Al'
        return struct

    def D03(self, a=None):
        struct = self.BCC(a=a).repeat(2)
        manually_identified_Al_sites = [5, 9, 3, 15]
        struct[manually_identified_Al_sites] = 'Al'
        return struct

    @property
    def D03_fractions(self):
        """Fractions of the unit cell for each unique site type, aFe, bFe, and Al."""
        return _D03Fractions()

    def random_BCC(self, a=None, repeat=2, c_Al=None):
        c_Al = self._c_Al if c_Al is None else c_Al
        struct = self.BCC(a=a).repeat(repeat)
        n_Al = round(c_Al * len(struct))
        struct[np.random.choice(range(len(struct)), n_Al, replace=False)] = 'Al'
        return struct

    def _D03_antisite_ids(self, structure, pre_swap_species, site_fraction):
        """Finds all symmetrically unique sites in the structure with the target species and site fraction."""
        equiv = structure.get_symmetry()['equivalent_atoms']
        unique, counts = np.unique(equiv, return_counts=True)

        sym = structure.get_chemical_symbols()
        pre_swap_types = unique[sym[unique] == pre_swap_species]
        site_type = pre_swap_types[np.argmin([np.abs(np.sum(equiv == i) - site_fraction) for i in pre_swap_types])]

        return np.arange(len(structure))[equiv == site_type]

    def _D03_antisite_fraction_to_count(self, structure, site_fraction, c_antisites):
        return round(c_antisites * site_fraction * len(structure)) if c_antisites is not None else 1

    def _random_D03_antisites(self, from_species, to_species, site_fraction, a=None, repeat=1, c_antisites=None):
        structure = self.D03(a=a).repeat(repeat)
        n_antisites = self._D03_antisite_fraction_to_count(structure, site_fraction, c_antisites)
        available_antisite_ids = self._D03_antisite_ids(structure, from_species, site_fraction)
        structure[np.random.choice(available_antisite_ids, n_antisites, replace=False)] = to_species
        return structure

    def random_D03_antisites_Al_to_Fe(self, a=None, repeat=1, c_antisites=None):
        return self._random_D03_antisites(
            'Al', 'Fe', self.D03_fractions.Al, a=a, repeat=repeat, c_antisites=c_antisites
        )

    def random_D03_antisites_aFe_to_Al(self, a=None, repeat=1, c_antisites=None):
        return self._random_D03_antisites(
            'Fe', 'Al', self.D03_fractions.aFe, a=a, repeat=repeat, c_antisites=c_antisites
        )

    def random_D03_antisites_bFe_to_Al(self, a=None, repeat=1, c_antisites=None):
        return self._random_D03_antisites(
            'Fe', 'Al', self.D03_fractions.bFe, a=a, repeat=repeat, c_antisites=c_antisites
        )

    def FCC(self, a=None):
        return self._factory.bulk(
            'Fe',
            crystalstructure='fcc',
            a=a if a is not None else self._fcc_lattice_constant,
            cubic=True
        )

    @property
    def _fcc_lattice_constant(self):
        d_1NN = self.BCC().get_neighbors(num_neighbors=1, id_list=[0]).distances[0, 0]
        return d_1NN * np.sqrt(2)

    def random_FCC(self, a=None, repeat=2, c_Al=None):
        c_Al = self._c_Al if c_Al is None else c_Al
        struct = self.FCC(a=a).repeat(repeat)
        n_Al = round(c_Al * len(struct))
        struct[np.random.choice(range(len(struct)), n_Al, replace=False)] = 'Al'
        return struct


class _D03Fractions:
    @property
    def Al(self):
        return 0.25

    @property
    def aFe(self):
        return 0.5

    @property
    def bFe(self):
        return 0.25
