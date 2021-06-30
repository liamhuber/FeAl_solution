# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

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

    @self_if_arg_is_none
    def append(self, other):
        return JobName(super(JobName, self).__add__('_' + self._filter_string(other)))

    @self_if_arg_is_none
    def T(self, temperature, ndigits=2):
        """Temperature."""
        return self.append(f'{round(temperature, ndigits=ndigits)}K')

    @self_if_arg_is_none
    def potl(self, potl_index):
        return self.append(f'potl{potl_index}')

    @self_if_arg_is_none
    def c_Al(self, c_Al, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cAl{round(c_Al * 100, ndigits=ndigits)}')

    @self_if_arg_is_none
    def c_D03_anti_Al_to_Fe(self, c_antisites, ndigits=2):
        """Given Al atomic fraction, gives name with Al atomic percentage."""
        return self.append(f'cDAl2Fe{round(c_antisites * 100, ndigits=ndigits)}')

    @self_if_arg_is_none
    def repeat(self, n_reps):
        """Cell repetition (integer only)."""
        return self.append(f'rep{n_reps}')

    @self_if_arg_is_none
    def trial(self, trial):
        """Stochastic trial repetition."""
        return self.append(f'trl{trial}')

    @self_if_arg_is_none
    def a(self, a, ndigits=2):
        """Lattice constant."""
        return self.append(f'a{round(a, ndigits=ndigits)}')

    @self_if_arg_is_none
    def P(self, pressure, ndigits=2):
        """Pressure."""
        return self.append(f'P{round(pressure, ndigits=ndigits)}')

    @property
    def BCC(self):
        return self.append('bcc')

    @property
    def FCC(self):
        return self.append('fcc')

    @property
    def random_BCC(self):
        return self.append('rbcc')

    @property
    def B2(self):
        return self.append('b2')

    @property
    def D03(self):
        return self.append('d03')


class HasProject:
    def __init__(self, project):
        self._project = project

    @property
    def project(self):
        return self._project
