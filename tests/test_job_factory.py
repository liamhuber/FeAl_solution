# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal._test import TestWithProject
from pyiron_feal.factories.job import JobFactory
import numpy as np


class TestJobFactory(TestWithProject):

    def setUp(self):
        super().setUp()
        self.jf = JobFactory(self.project)

    def test_minimize(self):
        min_ = self.jf.minimize

        job = min_.BCC(0, a=5, repeat=2)
        self.assertEqual(10, job.structure.cell[0, 0])

        job = min_.FCC()
        for xl, count in job.structure.analyse.pyscal_cna_adaptive().items():
            if xl == 'fcc':
                self.assertEqual(count, len(job.structure))
            else:
                self.assertEqual(count, 0)

        job = min_.random_BCC(Al_at_frac=(1/8))
        self.assertAlmostEqual((1/8), np.sum(job.structure.get_chemical_symbols() == 'Al') / len(job.structure))

        job = min_.B2()
        sym = job.structure.get_chemical_symbols()
        self.assertEqual(np.sum(sym == 'Al'), np.sum(sym == 'Fe'))

        job = min_.D03()
        self.assertAlmostEqual(0.25, np.sum(job.structure.get_chemical_symbols() == 'Al') / len(job.structure))
