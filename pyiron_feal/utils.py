# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

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


def self_if_arg_is_none(fnc):
    def wrapper(self, arg, **kwargs):
        if arg is None:
            return self
        return fnc(self, arg, **kwargs)
    return wrapper


class JobName(str):
    @staticmethod
    def _filter_string(val):
        return str(val).replace('.', '_').replace('-', 'm')

    def __new__(cls, val):
        return super().__new__(cls, cls._filter_string(val))

    @property
    def string(self):
        return str(self)

    def __call__(
            self,
            interactive=False,
            potl_index=None,
            bcc=False,
            d03=False,
            b2=False,
            fcc=False,
            columnar=False,
            a=None,
            repeat=None,
            trial=None,
            pressure=None,
            temperature=None,
            c_Al=None,
            c_D03_anti_Al_to_Fe=None,
            c_D03_anti_aFe_to_Al=None,
            c_D03_anti_bFe_to_Al=None,
            c_B2_anti_Al_to_Fe=None,
            c_B2_anti_Fe_to_Al=None,
            max_cluster_fraction=None,
            symbol_ref=None,
            ndigits=2
    ):
        if interactive:
            self = self.interactive
        self = self.potl(potl_index)
        if bcc:
            self = self.bcc
        if d03:
            self = self.d03
        if b2:
            self = self.b2
        if fcc:
            self = self.fcc
        if columnar:
            self = self.columnar
        self = self.a(a, ndigits=ndigits)
        self = self.repeat(repeat)
        self = self.trial(trial)
        self = self.T(temperature, ndigits=ndigits)
        self = self.P(pressure, ndigits=ndigits)
        self = self.c_Al(c_Al, ndigits=ndigits)
        self = self.c_D03_anti_Al_to_Fe(c_D03_anti_Al_to_Fe, ndigits=ndigits)
        self = self.c_D03_anti_aFe_to_Al(c_D03_anti_aFe_to_Al, ndigits=ndigits)
        self = self.c_D03_anti_bFe_to_Al(c_D03_anti_bFe_to_Al, ndigits=ndigits)
        self = self.c_B2_anti_Al_to_Fe(c_B2_anti_Al_to_Fe, ndigits=ndigits)
        self = self.c_B2_anti_Fe_to_Al(c_B2_anti_Fe_to_Al, ndigits=ndigits)
        self = self.max_cluster_fraction(max_cluster_fraction, ndigits=ndigits)
        self = self.symbol_ref(symbol_ref)
        return self.string

    @self_if_arg_is_none
    def append(self, other):
        return JobName(super(JobName, self).__add__('_' + self._filter_string(other)))

    @property
    def interactive(self):
        return self.append('i')

    @self_if_arg_is_none
    def potl(self, potl_index):
        return self.append(f'p{potl_index}')

    @property
    def bcc(self):
        return self.append('bcc')\

    @property
    def d03(self):
        return self.append('d03')

    @property
    def b2(self):
        return self.append('b2')

    @property
    def fcc(self):
        return self.append('fcc')

    @property
    def columnar(self):
        return self.append('columnar')

    @self_if_arg_is_none
    def a(self, a, ndigits=2):
        """Lattice constant."""
        return self.append(f'a{round(a, ndigits=ndigits)}')

    @self_if_arg_is_none
    def repeat(self, n_reps):
        """Cell repetition (integer only)."""
        return self.append(f'r{n_reps}')

    @self_if_arg_is_none
    def trial(self, trial):
        """Stochastic trial repetition."""
        return self.append(f't{trial}')

    @self_if_arg_is_none
    def T(self, temperature, ndigits=2):
        """Temperature."""
        return self.append(f'{round(temperature, ndigits=ndigits)}K')

    @self_if_arg_is_none
    def P(self, pressure, ndigits=2):
        """Pressure."""
        return self.append(f'P{round(pressure, ndigits=ndigits)}')

    def _concentration(self, c, ndigits=2):
        try:
            return round(c * 100, ndigits=ndigits)
        except TypeError:
            return c.lower()[:3]

    @self_if_arg_is_none
    def c_Al(self, c_Al, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cAl{self._concentration(c_Al, ndigits=ndigits)}')

    @self_if_arg_is_none
    def c_D03_anti_Al_to_Fe(self, c_antisites, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cDAl{self._concentration(c_antisites, ndigits=ndigits)}')

    @self_if_arg_is_none
    def c_D03_anti_aFe_to_Al(self, c_antisites, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cDaFe{self._concentration(c_antisites, ndigits=ndigits)}')

    @self_if_arg_is_none
    def c_D03_anti_bFe_to_Al(self, c_antisites, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cDbFe{self._concentration(c_antisites, ndigits=ndigits)}')

    @self_if_arg_is_none
    def c_B2_anti_Al_to_Fe(self, c_antisites, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cBAl{self._concentration(c_antisites, ndigits=ndigits)}')

    @self_if_arg_is_none
    def c_B2_anti_Fe_to_Al(self, c_antisites, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cBFe{self._concentration(c_antisites, ndigits=ndigits)}')

    @self_if_arg_is_none
    def max_cluster_fraction(self, fraction, ndigits=2):
        return self.append(f'cf{round(fraction, ndigits=ndigits)}')

    @self_if_arg_is_none
    def symbol_ref(self, name):
        return self.append(f'r{name}')


class HasProject:
    def __init__(self, project):
        self._project = project

    @property
    def project(self):
        return self._project


def bfs(node, topology, condition_fnc, **condition_kwargs):
    """
    Breadth first search building a cluster starting at one node and obeying a condition function for adding new nodes.

    Args:
        node (int): Which node to start the search from.
        topology (numpy.ndarray | list): Per-site list of lists giving the all neighbouring nodes (i.e. a
            `neighbors.indices` object).
        condition_fnc (fnc): A function for evaluating whether or not connected nodes should be added.
        *condition_args: Additional arguments for the condition function.

    Returns:
        (numpy.ndarray): The cluster built from the requested node obeying the condition function.
    """
    cluster = [node]
    queue = [node]
    while queue:
        i = queue.pop()
        to_add = [j for j in topology[i] if condition_fnc(i, j, **condition_kwargs)]
        queue += np.setdiff1d(to_add, cluster).astype(int).tolist()
        cluster = np.unique(cluster + to_add).astype(int).tolist()

    return np.array(cluster)
