# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from unittest import TestCase
from pyiron_feal.utils import JobName, bfs
import numpy as np


class TestJobName(TestCase):

    def test_filter(self):
        self.assertEqual('foo_mbar', JobName._filter_string('foo.-bar'), msg="Unexpected replacements.")

    def test_init(self):
        self.assertEqual('foo_mbar', JobName('foo.-bar'))

    def test_add(self):
        self.assertEqual('foobar', JobName('foo') + 'bar')
        self.assertIsInstance(JobName('foo') + 'bar', str, msg="Did you override __add__?")

    def test_append(self):
        self.assertEqual('foo_bar', JobName('foo').append('bar'))
        self.assertIsInstance(JobName('foo').append('bar'), JobName,
                              msg="Did you accidentally return a regular string?")

    def test_temperature(self):
        self.assertEqual('foo_273K', JobName('foo').T(273))
        self.assertEqual('foo_273_0K', JobName('foo').T(273.0))

    def test_other_tags(self):
        self.assertEqual('foo_potl42', JobName('foo').potl(42))
        self.assertEqual('foo_cAl33_33', JobName('foo').c_Al(0.33333333))
        self.assertEqual('foo_cAl33_333', JobName('foo').c_Al(0.33333333, ndigits=3))
        self.assertEqual('foo_rep4', JobName('foo').repeat(4))
        self.assertEqual('foo', JobName('foo').repeat(None))
        self.assertEqual('foo_trl3', JobName('foo').trial(3))
        self.assertEqual('foo_bcc', JobName('foo').bcc)
        self.assertEqual('foo_fcc', JobName('foo').fcc)
        self.assertEqual('foo_b2', JobName('foo').b2)
        self.assertEqual('foo_d03', JobName('foo').d03)
        self.assertEqual('foo_P0_0', JobName('foo').P(0.))
        self.assertEqual('foo_a4_36', JobName('foo').a(4.355))
        self.assertEqual('foo_cDAl2Fedil', JobName('foo').c_D03_anti_Al_to_Fe('Dilute'))

    def test_call(self):
        name = JobName('foo')
        self.assertEqual(
            'foo_potl42_bcc_rep4_trl3_273K_P0_0_cDAl2Fe11_1',
            name(
                potl_index=42,
                bcc=True,
                repeat=4,
                trial=3,
                temperature=273,
                pressure=0.,
                c_D03_anti_Al_to_Fe=0.1111111,
                ndigits=1
            )
        )
        self.assertEqual('foo', name.string, msg="Calling shouldn't overwrite the base object.")


class TestBFS(TestCase):

    def test_bfs(self):
        """
        0  1  2  3
        4  5  6  7
        8  9  10 11
        12 13 14 15

        x x x o
        x x o o
        x x x o
        x x x o
        """
        l = 4
        nodes = np.arange(l*l, dtype=int)
        nodegrid = np.reshape(nodes, (l, l))
        topology = np.stack(
            (
                np.roll(nodegrid, 1, axis=1),
                np.roll(nodegrid, -1, axis=1),
                np.roll(nodegrid, 1, axis=0),
                np.roll(nodegrid, -1, axis=0),
            ),
            axis=-1
        ).reshape(l*l, -1)

        signature = np.array('x x x o x x o o x x x o x x x o'.split())

        def condition(i, j, topo, sig, thresh):
            return (sig[i] == sig[j]) and (np.sum(sig[topo[j]] == sig[j]) >= thresh)

        self.assertCountEqual(
            [9, 13, 1],
            bfs(9, topology, condition, topo=topology, sig=signature, thresh=4).tolist(),
            msg="Should only get x's completely surrounded by x's. Don't forget we have periodic boundary conditions."
        )
        self.assertCountEqual(
            nodes[signature == 'x'].tolist(),
            bfs(9, topology, condition, topo=topology, sig=signature, thresh=0).tolist(),
            msg="With no threshold, should get all nodes with the same signature."
        )
        self.assertCountEqual(
            [3, 7, 11, 15],
            bfs(7, topology, condition, topo=topology, sig=signature, thresh=2).tolist(),
            msg="Should be getting that righthand column, minus the nub at 6."
        )