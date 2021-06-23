# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal._test import TestWithProject
from pyiron_feal.factories.job import JobFactory


class TestJobFactory(TestWithProject):

    def setUp(self):
        super().setUp()
        self.jf = JobFactory(self.project)

    def test_minimize(self):
        job = self.jf.minimize.bcc(0, a=5, repeat=2)
        self.assertEqual(10, job.structure.cell[0, 0])