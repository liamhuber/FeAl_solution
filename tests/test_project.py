# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal._test import TestWithProject


class TestProject(TestWithProject):

    def test_input(self):
        self.assertListEqual(
            self.project.input.potentials,
            self.project.input.potentials_eam + self.project.input.potentials_meam
        )
