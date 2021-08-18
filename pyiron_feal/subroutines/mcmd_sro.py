# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal.utils import HasProject, bfs
import numpy as np
from functools import lru_cache
from pyiron_base import GenericJob
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]
import matplotlib.pyplot as plt
import seaborn as sns

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


def _environment_matches(site, e1, e2, topo, thresh=None):
    """
    Checks if the site has matching surroundings in both environments.

    Args:
        site (int): Which site to check.
        e1 (numpy.ndarray): A per-site description of environment 1, e.g. the chemical species.
        e2 (numpy.ndarray): A per-site description of the reference environment, e.g. the chemical species.
        topo (numpy.ndarray | list): Per-site list of lists giving the all neighbouring sites (i.e. a
            `neighbors.indices` object).
        thresh (int | None): If an integer, return true when at least that number of site neighbors identified by the
            topology match between the test environment and reference environments. (Default is None, which requires
            *all* neighbors to match.)

    Returns:
        (bool): Whether or not the two environments match.
    """
    neighbors = topo[site]
    if thresh is None:
        return (e1[site] == e2[site]) and np.all(e1[neighbors] == e2[neighbors])
    else:
        return (e1[site] == e2[site]) and np.sum(e1[neighbors] == e2[neighbors]) >= thresh


def _condition(_, site, e1, e2, topo, thresh=None):
    return _environment_matches(site, e1, e2, topo, thresh=thresh)


class _Cluster:
    """
    A class for building clusters of like-environments.

    Attributes:
        topology (numpy.ndarray | list): Per-site list of lists giving the all neighbouring sites (i.e. a
                `neighbors.indices` object).
        reference_environments (dict): A dictionary of per-site descriptions of environment against which to check for
            similarity, e.g. the chemical species.
        threshold (int | None): If an integer, return true when at least that number of site neighbors identified by the
            topology match between the test environment and reference environments. (Default is None, which requires
            *all* neighbors to match.)
    """
    def __init__(self, topology, reference_environments, threshold=None):
        self.topology = topology
        self.reference_environments = reference_environments
        self.threshold = threshold

    def _get_matching_sites(self, env, ref_env):
        """
        Finds all sites with matching surroundings in both environments

        Args:
            env (numpy.ndarray): A per-site description of the environment to test, e.g. the chemical species.
            ref_env (numpy.ndarray): A per-site description of to environment to test against, e.g. the chemical
                species.

        Returns:
            (numpy.ndarray): The indices for each site where the two environments match.
        """
        return np.array([
            i for i in np.arange(len(env))
            if _environment_matches(i, env, ref_env, self.topology, thresh=self.threshold)
        ])

    def _get_clusters(self, env, ref_env):
        """
        Use breadth-first-search to build all clusters of sites in environment 1 that are the same as in environment 2
        *and* have (up to a given threshold) the same local environment.

        Args:
            env (numpy.ndarray): A per-site description of the environment to test, e.g. the chemical species.
            ref_env (numpy.ndarray): A per-site description of to environment to test against, e.g. the
                chemical species.

        Returns:
            (list): The nested list of cluster IDs.
        """
        matches = self._get_matching_sites(env, ref_env)

        clusters = []
        while len(matches) > 0:
            i = matches[-1]
            matches = matches[:-1]
            clusters.append(
                bfs(i, self.topology, _condition, topo=self.topology, e1=env, e2=ref_env, thresh=self.threshold)
            )
            matches = np.setdiff1d(matches, clusters[-1])

        return clusters

    def get_clusters(self, env):
        return {k: self._get_clusters(env, v) for k, v in self.reference_environments.items()}

    def __call__(self, env, threshold=np.nan):
        if not np.isnan(threshold):
            self.threshold = threshold
        return self.get_clusters(env)


class MCMDSRO(HasProject):

    def __init__(self, project):
        super().__init__(project)
        self._cluster = None

    @property
    def cluster(self):
        if self._cluster is None:
            raise ValueError('First run define_clustering')
        return self._cluster

    def define_clustering(self, topology, reference_environments, threshold=None):
        """
        Sets the `cluster` method to build clusters according to the provided topology and references.
        Args:
            topology (numpy.ndarray | list): Per-site list of lists giving the all neighbouring sites (i.e. a
                    `neighbors.indices` object).
            reference_environments (dict): A dictionary of per-site descriptions of environment against which to check for
                similarity, e.g. the chemical species.
            threshold (int | None): If an integer, return true when at least that number of site neighbors identified by the
                topology match between the test environment and reference environments. (Default is None, which requires
                *all* neighbors to match.)
        """
        self._cluster = _Cluster(topology, reference_environments, threshold=threshold)
