from unittest import TestCase
from pyiron_feal.factories.structure import StructureFactory
import numpy as np


class TestStructureFactory(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sf = StructureFactory().FeAl

    def test_cell(self):
        bcc = self.sf.BCC()
        cell = bcc.cell.array
        self.assertTrue(np.allclose(np.diag(cell), cell[0, 0]), msg="Cell not cubic")
        double = self.sf.BCC(a=2*cell[0, 0])
        self.assertAlmostEqual((2**3) * bcc.get_volume(), double.get_volume(), msg=" correctly")

    def test_lengths_the_same(self):
        n_bcc = len(self.sf.BCC())
        n_b2 = len(self.sf.B2())
        n_d03 = len(self.sf.D03())
        n_fcc = len(self.sf.FCC())
        self.assertEqual(n_bcc, n_b2, msg="B2 structure not the same length as BCC structure")
        self.assertEqual((2**3)*n_bcc, n_d03, msg="D03 structure not the same length as BCC structure")
        self.assertEqual(2*n_bcc, n_fcc,
                         msg="FCC structure has twice as many atoms in unit cell, so should be 2x larger")

    @staticmethod
    def _get_frac_Al(structure):
        return np.sum(structure.get_chemical_symbols() == 'Al') / len(structure)

    def test_get_frac_Al(self):
        struct = self.sf.BCC()
        struct[0] = 'Al'
        struct[1:] = 'Fe'
        self.assertAlmostEqual(1. / len(struct), self._get_frac_Al(struct))

    def test_B2(self):
        self.assertAlmostEqual(0.5, self._get_frac_Al(self.sf.B2()), msg="B2 chemistry incorrect")
        # Lazily ignoring the location of the Al atoms

    def test_D03(self):
        self.assertAlmostEqual(0.25, self._get_frac_Al(self.sf.D03()), msg="D03 chemistry incorrect")
        # Lazily ignoring the location of the Al atoms

    def test_random(self):
        random = self.sf.random_BCC()
        self.assertAlmostEqual(
            1,
            self.sf._Al_at_frac / self._get_frac_Al(random),
            places=1,
            msg=f"Fraction Al {self._get_frac_Al(random)} was not within 10% of target {self.sf._Al_at_frac}."
        )

    def test_random_fcc(self):
        random = self.sf.random_FCC(repeat=3)
        self.assertAlmostEqual(
            1,
            self.sf._Al_at_frac / self._get_frac_Al(random),
            places=1,
            msg=f"Fraction Al {self._get_frac_Al(random)} was not within 10% of target {self.sf._Al_at_frac}."
        )
