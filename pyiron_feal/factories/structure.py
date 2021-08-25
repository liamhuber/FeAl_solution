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

    def _double_unit(self, a=None):
        return self._factory.bulk('Fe', a=a, cubic=True).repeat(2)

    @staticmethod
    def _random_species_change(structure, valid_sites, concentration, new_symbol):
        if isinstance(concentration, str) and concentration.lower() == 'dilute':
            concentration = 1 / len(valid_sites)
        elif concentration is None or np.isclose(concentration, 0):
            return structure

        n = len(valid_sites)
        n_swaps = min(round(concentration * n), n)
        structure[np.random.choice(valid_sites, n_swaps, replace=False)] = new_symbol
        return structure

    def bcc(self, a=None, repeat=1, c_Al=None):
        structure = self._double_unit(a=a).repeat(repeat)
        structure = self._random_species_change(structure, np.arange(len(structure)), c_Al, 'Al')
        return structure

    @property
    def d03_fractions(self):
        """Fractions of the unit cell for each unique site type, aFe, bFe, and Al."""
        return _D03Fractions()

    def _d03_antisite_ids(self, structure, pre_swap_species, site_fraction):
        """Finds all symmetrically unique sites in the structure with the target species and site fraction."""
        equiv = structure.get_symmetry()['equivalent_atoms']
        unique, counts = np.unique(equiv, return_counts=True)

        sym = structure.get_chemical_symbols()
        pre_swap_types = unique[sym[unique] == pre_swap_species]
        site_type = pre_swap_types[np.argmin([np.abs(np.mean(equiv == i) - site_fraction) for i in pre_swap_types])]

        return np.arange(len(structure))[equiv == site_type]

    def d03(
            self,
            a=None,
            repeat=1,
            c_D03_anti_Al_to_Fe=None,
            c_D03_anti_aFe_to_Al=None,
            c_D03_anti_bFe_to_Al=None,
            basis=0
    ):
        structure = self._double_unit(a=a)
        Al_site_bases = [
            [3, 5, 9, 15],
            [1, 7, 11, 13],
            [0, 6, 10, 12],
            [2, 4, 8, 14]
        ]
        structure[Al_site_bases[basis]] = 'Al'
        structure = structure.repeat(repeat)
        Al_ids = self._d03_antisite_ids(structure, 'Al', self.d03_fractions.Al)
        aFe_ids = self._d03_antisite_ids(structure, 'Fe', self.d03_fractions.aFe)
        bFe_ids = self._d03_antisite_ids(structure, 'Fe', self.d03_fractions.bFe)
        structure = self._random_species_change(structure, Al_ids, c_D03_anti_Al_to_Fe, 'Fe')
        structure = self._random_species_change(structure, aFe_ids, c_D03_anti_aFe_to_Al, 'Al')
        structure = self._random_species_change(structure, bFe_ids, c_D03_anti_bFe_to_Al, 'Al')
        return structure

    def b2(
            self,
            a=None,
            repeat=1,
            c_B2_anti_Al_to_Fe=None,
            c_B2_anti_Fe_to_Al=None,
            basis=1
    ):
        structure = self._double_unit(a=a)
        structure[np.arange(basis, len(structure), 2, dtype=int)] = 'Al'
        structure = structure.repeat(repeat)
        half_the_sites = np.arange(0, len(structure), 2, dtype='int')
        structure = self._random_species_change(structure, half_the_sites, c_B2_anti_Fe_to_Al, 'Al')
        structure = self._random_species_change(structure, half_the_sites + 1, c_B2_anti_Al_to_Fe, 'Fe')
        return structure

    @property
    def _fcc_lattice_constant(self):
        d_1nn = self._double_unit().get_neighbors(num_neighbors=1, id_list=[0]).distances[0, 0]
        return d_1nn * np.sqrt(2)

    def fcc(self, a=None, repeat=1, c_Al=None):
        structure = self._factory.bulk(
            'Fe',
            crystalstructure='fcc',
            a=a if a is not None else self._fcc_lattice_constant,
            cubic=True
        ).repeat(int(2 * repeat))
        structure = self._random_species_change(structure, np.arange(len(structure)), c_Al, 'Al')
        return structure

    def columnar_b2(self, planar_repeats=1):
        structure = self._factory.bulk('Fe', cubic=True).repeat((planar_repeats, planar_repeats, 1))
        structure[0] = 'Al'
        return structure


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
