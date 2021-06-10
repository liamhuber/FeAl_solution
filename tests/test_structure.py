from unittest import TestCase

import numpy as np

from pyiron_feal.structure_factory import StructureFactory


class TestStructureFactory(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sf = StructureFactory().FeAl

    def test_cell(self):
        bcc = self.sf.BCC()
        cell = bcc.cell.array
        self.assertTrue(np.allclose(np.diag(cell), cell[0, 0]), msg="Cell not cubic")
        a = cell[0, 0] / 2  # Since bcc cell is repeated by default
        double = self.sf.BCC(a=2*a)
        self.assertAlmostEqual((2**3) * bcc.get_volume(), double.get_volume(), msg=" correctly")

    def test_lengths_the_same(self):
        n_bcc = len(self.sf.BCC())
        n_b2 = len(self.sf.B2())
        n_d03 = len(self.sf.D03())
        self.assertEqual(n_b2, n_bcc, msg="B2 structure not the same length as BCC structure")
        self.assertEqual(n_d03, n_bcc, msg="D03 structure not the same length as BCC structure")

    @staticmethod
    def _get_frac_Al(structure):
        return np.sum(structure.get_chemical_symbols() == 'Al') / len(structure)

    def test_B2(self):
        self.assertAlmostEqual(0.5, self._get_frac_Al(self.sf.B2()), msg="B2 chemistry incorrect")
        # Lazily ignoring the location of the Al atoms

    def test_D03(self):
        self.assertAlmostEqual(0.25, self._get_frac_Al(self.sf.D03()), msg="D03 chemistry incorrect")
        # Lazily ignoring the location of the Al atoms
