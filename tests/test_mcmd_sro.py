# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from unittest import TestCase
from pyiron_feal._test import TestWithProject
from pyiron_feal.subroutines.mcmd_sro import _environment_matches, _Cluster, MCMDSRO
import numpy as np


env1 = np.array(['x', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'x'])
env2 = np.array(['x', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o'])
ids = np.arange(len(env1), dtype=int)
topology = np.stack((np.roll(ids, 1), np.roll(ids, -1))).T


class TestEnvironmentMatch(TestCase):

    def test_environment_matches(self):
        self.assertTrue(_environment_matches(1, env1, env2, topology))
        self.assertTrue(_environment_matches(1, env1, env2, topology, thresh=1))
        self.assertFalse(_environment_matches(3, env1, env2, topology))
        self.assertTrue(_environment_matches(3, env1, env2, topology, thresh=1))
        self.assertFalse(_environment_matches(8, env1, env2, topology, thresh=1))


class TestCluster(TestCase):
    def setUp(self) -> None:
        self.clust = _Cluster(topology, {'env2': env2})

    def test_get_matching_sites(self):
        self.assertListEqual([1, 2, 6], self.clust._get_matching_sites(env1, env2).tolist())

        self.clust.threshold = 2
        self.assertListEqual([1, 2, 6], self.clust._get_matching_sites(env1, env2).tolist())

        self.clust.threshold = 1
        self.assertListEqual(
            [0, 1, 2, 3, 5, 6, 7],
            self.clust._get_matching_sites(env1, env2).tolist()
        )

    def test_get_clusters(self):
        for ref, calc in zip([[6], [1, 2]], self.clust._get_clusters(env1, env2)):
            self.assertListEqual(ref, calc.tolist())
            # TODO: Find a test that is order-agnostic, the order here is not actually important...


class TestJobFactory(TestWithProject):

    def setUp(self):
        super().setUp()
        self.mcmdsro = MCMDSRO(self.project)

    def test_clustering(self):
        with self.assertRaises(ValueError):
            self.mcmdsro.cluster
        self.mcmdsro.define_clustering(topology, {'env2': env2})
        for ref, calc in zip([[6], [1, 2]], self.mcmdsro.cluster(env1)['env2']):
            self.assertListEqual(ref, calc.tolist())
