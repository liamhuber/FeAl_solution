from unittest import TestCase
from pyiron_feal.factories.structure import StructureFactory
import numpy as np


class TestStructureFactory(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sf = StructureFactory().FeAl

    def test_cell(self):
        bcc = self.sf.bcc()
        cell = bcc.cell.array
        self.assertTrue(np.allclose(np.diag(cell), cell[0, 0]), msg="Cell not cubic")
        double = self.sf.bcc(a=2*cell[0, 0])
        self.assertAlmostEqual(
            (2**3) * (2**3) * bcc.get_volume(), double.get_volume(),
            msg="Double lattice parameter not giving (double lattice)*(two repeats of BCC unit) times more volume"
        )

    def test_lengths_the_same(self):
        n_bcc = len(self.sf.bcc())
        n_b2 = len(self.sf.b2())
        n_d03 = len(self.sf.d03())
        n_fcc = len(self.sf.fcc())
        self.assertEqual(n_bcc, n_b2, msg="B2 structure not the same length as BCC structure")
        self.assertEqual(n_bcc, n_d03, msg="D03 structure not the same length as BCC structure")
        self.assertEqual(2*n_bcc, n_fcc,
                         msg="FCC structure has twice as many atoms in unit cell, so should be 2x atom count")

    @staticmethod
    def _get_frac_Al(structure):
        return np.sum(structure.get_chemical_symbols() == 'Al') / len(structure)

    @staticmethod
    def _count_Al(structure):
        return np.sum(structure.get_chemical_symbols() == 'Al')

    def test_get_frac_Al(self):
        n_bcc = len(self.sf.bcc())
        struct = self.sf.bcc(c_Al=1/n_bcc)
        self.assertAlmostEqual(1. / len(struct), self._get_frac_Al(struct))

    def test_bcc(self):
        n_bcc = len(self.sf.bcc())
        self.assertEqual(1, self._count_Al(self.sf.bcc(c_Al=1 / n_bcc)))
        self.assertEqual(n_bcc, self._count_Al(self.sf.bcc(c_Al=1)))

    def test_b2(self):
        self.assertAlmostEqual(0.5, self._get_frac_Al(self.sf.b2()), msg="B2 chemistry incorrect")
        # Lazily ignoring the location of the Al atoms

    def test_d03(self):
        self.assertAlmostEqual(0.25, self._get_frac_Al(self.sf.d03()), msg="D03 chemistry incorrect")
        # Lazily ignoring the location of the Al atoms

    def test_fcc(self):
        structure = self.sf.fcc(c_Al=0.5)
        self.assertAlmostEqual(0.5 * len(structure), self._count_Al(structure), msg="Solid solution chemistry wrong")
