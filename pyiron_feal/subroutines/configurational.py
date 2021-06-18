# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal.utils import HasProject, JobName
import numpy as np
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

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


class Configurational(HasProject):
    pass


class _EnergyCalculator(HasProject):

    def __init__(self, project, E_BCC=None, E_form=None, E_B2=None, E_D03=None):
        """E's should be per-atom energies. Defaults are scraped from a notebook for the Mendelev potential"""
        super().__init__(project)
        self.E_BCC = E_BCC or (-8.025965/2)
        self.E_form = E_form or (-64.749742 - 16*self.E_BCC)/3
        self.E_B2 = E_B2 or (-8.049821/2)
        self.E_D03 = E_D03 or (-64.676110/16)

    def G_SS(self, c, T):
        return self.E_BCC + c * self.E_form + KB * T * ((1 - c) * np.log(1 - c) + c * np.log(c))

    def dmu(self, c, T):
        """dG_SS/dc"""
        return self.E_form + KB * T * np.log(c / (1 - c))

    def G_B2(self, c, T):
        return self.E_B2 + (0.5 - c) * (-self.dmu(c, T))

    def G_D03(self, c, T):
        return self.E_D03 + (0.25 - c) * (-self.dmu(c, T))

    def X(self, G, T):
        return np.exp(-G / (KB * T))

    def partition(self, c, T, c_nominal=0.18):
        w_SS = np.exp(-self.G_SS(c, T) / (KB * T))
        w_B2 = np.exp(-self.G_B2(c, T) / (KB * T))
        w_D03 = np.exp(-self.G_D03(c, T) / (KB * T))
        Z = w_SS + w_B2 + w_D03
        X_SS = w_SS / Z
        X_B2 = w_B2 / Z
        X_D03 = w_D03 / Z
        c_SS = (c_nominal - 0.5 * X_B2 - 0.25 * X_D03) / X_SS
        if np.any(c_SS < 0):
            raise ValueError('Calculated solid solution concentration was negative.')
        return c_SS, np.array([X_SS, X_B2, X_D03])

    def equilibrate(self, T, c_nominal=0.18, fraction_change_threshold=1e-3, max_iter=10000):
        c_SS = c_nominal
        X_last = np.inf
        for _ in np.arange(max_iter):
            c_SS, X = self.partition(c_SS, T, c_nominal=c_nominal)
            max_diff = np.abs(X - X_last).max()
            if max_diff < fraction_change_threshold:
                return c_SS, X
            X_last = X
        raise RuntimeError('Reached max steps without converging')

    def G_vec(self, c, T):
        return self.G_SS(c, T), self.G_B2(c, T), self.G_D03(c, T)

    def G(self, c, T, c_nominal=0.18):
        """Warning: figure this out"""
        c, X = self.partition(c, T, c_nominal=c_nominal)
        G = self.G_vec(c, T)
        return np.sum(X*G)

    def plot_free_energies(self, concentrations, temperature):
        Gs = self.G_vec(concentrations, temperature)
        plt.plot(concentrations, Gs[0], label='SS')
        plt.plot(concentrations, Gs[1], label='B2')
        plt.plot(concentrations, Gs[2], label='D03')
        plt.xlabel('c_SS')
        plt.ylabel('G [eV]')
        plt.legend()

    def build_phase_fractions(
            self,
            temperatures,
            nominal_concentrations,
            fraction_change_threshold=1e-3,
            max_iter=10000
    ):
        all_X = np.zeros((len(temperatures), len(nominal_concentrations), 3))
        ss_concentration = np.zeros((len(temperatures), len(nominal_concentrations)))
        for i, T in enumerate(temperatures):
            for j, c_nominal in enumerate(nominal_concentrations):
                try:
                    c_SS, X = self.equilibrate(
                        T,
                        c_nominal=c_nominal,
                        fraction_change_threshold=fraction_change_threshold,
                        max_iter=max_iter
                    )
                except ValueError:
                    c_SS = np.nan
                    X = np.array(3 * [np.nan])
                ss_concentration[i, j] = c_SS
                all_X[i, j] = X
        return ss_concentration, all_X

    def plot_concentrations(self, concentrations, extent):
        fig, ax = plt.subplots()
        con = ax.imshow(concentrations, extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel('Nominal Al concentration')
        ax.set_ylabel('$T$ [K]')
        cbar = plt.colorbar(con)
        ax.set_title("Solid solution Al concentration")
        return fig, ax

    @staticmethod
    def plot_phases(fractions, extent, cbar_loc='bottom', figsize=None, normalize_rgb=False):
        """
        Plot the phase fractions

        Args:
            fractions (numpy.ndarray (N, M, 3)): Phase fractions with final dimension in the order solid solution, B2,
                and D03
        """
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = fig.add_gridspec(4, 3)
        main = fig.add_subplot(gs[:-1, :])
        norm = np.nanmax(fractions, axis=(0, 1)) if normalize_rgb else np.ones(3)
        im = main.imshow(fractions / norm, extent=extent, aspect='auto', origin='lower')
        main.set_title('RGB Phase Map')
        main.set_xlabel('Nominal Al concentration')
        main.set_ylabel('$T$ [K]')
        for ax, label, cmap in zip(
                np.arange(3),
                ['SS', 'B2', 'D03'],
                ['Reds', 'Greens', 'Blues']
        ):
            sub = fig.add_subplot(gs[-1, ax])
            im = sub.imshow(fractions[..., ax], cmap=cmap, extent=extent, aspect='auto', origin='lower')
            sub.set_xlabel('$C_\mathrm{Al}^\mathrm{nom}$')
            sub.set_ylabel('$T$ [K]')
            sub.set_title(label)
            cbar = fig.colorbar(im, location=cbar_loc)
            cbar.ax.set_ylabel(f'$X_\mathrm{{{label}}}$')
        return fig, gs

    @staticmethod
    def extent_from_arrays(mu_array, temperature_array):
        return [mu_array.min(), mu_array.max(), temperature_array.min(), temperature_array.max()]

    def plot_fractions_1d(self, concentration_range, temperature_range, phase_fractions):
        i = np.argmin(abs(concentration_range - 0.18))
        pal = sns.color_palette("tab10")
        rgb = [pal[i] for i in [3, 2, 0]]
        expt = self.project.input.experimental_fractions
        fig, ax = plt.subplots()
        for n, color, label, style_ in zip(range(3), rgb, ['SS', 'B2', 'D03'], ['-', '--', ':']):
            ax.plot(temperature_range, phase_fractions[:, i, n], color=color, label=label, linestyle=style_)
            ax.scatter([expt['T']], [expt[label]], color=color, marker='o')
        ax.set_xlabel('Temeprature [K]')
        ax.set_ylabel('Phase fraction')
        ax.legend()
        return fig, ax
