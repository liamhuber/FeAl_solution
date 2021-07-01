# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal.utils import HasProject
import numpy as np
from functools import lru_cache
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


class ZeroK(HasProject):

    @staticmethod
    def _get_peratom_energy(job_creator, potl_index=0, run_again=False, **other_job_kwargs):
        job = job_creator(potl_index=potl_index, delete_existing_job=run_again, **other_job_kwargs)
        job.run()
        return job.output.energy_pot[-1] / len(job.structure)

    @lru_cache()
    def get_bcc_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.bcc,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_fcc_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.fcc,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_b2_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.b2,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_d03_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.D03,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    def plot_phases_0K(self, potl_index=0, ax=None, beautify=True):
        (fig, ax) = plt.subplots() if ax is None else (None, ax)
        e_fcc = self.get_fcc_peratom_energy(potl_index=potl_index)
        e_bcc = self.get_bcc_peratom_energy(potl_index=potl_index)
        e_d03 = self.get_d03_peratom_energy(potl_index=potl_index)
        e_b2 = self.get_b2_peratom_energy(potl_index=potl_index)

        c_Al = [0, 0, 0.25, 0.5]
        energies = [e_fcc, e_bcc, e_d03, e_b2]
        ax.plot(c_Al, energies, marker='o')
        if beautify:
            ax.set_xlabel('$c_\mathrm{Al}$')
            ax.set_ylabel('$E$ [eV/atom]')
            for c, E, label in zip(c_Al, energies, ['FCC', 'BCC', 'D03', 'B2']):
                ax.annotate(label, (c, E))
        return ax

    @staticmethod
    def _get_size_converged_point_defect_energy(
            ref_energy_per_atom, job_creator, stderr=1e-3, potl_index=0, run_again=False, **other_job_kwargs
    ):
        reps = 0
        actual_err = np.inf
        energies = [np.inf]
        while actual_err > stderr:
            reps += 1
            job = job_creator(
                potl_index=potl_index,
                repeat=reps,
                delete_existing_job=run_again,
                **other_job_kwargs
            )
            job.run()
            energies.append(job.output.energy_pot[-1] - len(job.structure) * ref_energy_per_atom)
            actual_err = np.abs(energies[-1] - energies[-2])
        return energies[-1], reps, actual_err

    @lru_cache()
    def get_dilute_formation_energy(self, stderr=1e-3, run_again=False, potl_index=0, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_bcc_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.bcc,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            c_Al='dilute',
            **other_job_kwargs
        )

    @lru_cache()
    def get_dilute_d03_Al_to_Fe_energy(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_d03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.d03,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            c_D03_anti_Al_to_Fe='dilute',
            **other_job_kwargs
        )

    @lru_cache()
    def get_dilute_d03_aFe_to_Al_energy(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_d03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.d03,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            c_D03_anti_aFe_to_Al='dilute',
            **other_job_kwargs
        )

    @lru_cache()
    def get_dilute_d03_bFe_to_Al_energy(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_d03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.d03,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            c_D03_anti_bFe_to_Al='dilute',
            **other_job_kwargs
        )

    @lru_cache()
    def get_dilute_b2_Al_to_Fe_energy(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_b2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.b2,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            c_B2_anti_Al_to_Fe='dilute',
            **other_job_kwargs
        )

    @lru_cache()
    def get_dilute_b2_Fe_to_Al_energy(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_b2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.b2,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            c_B2_anti_Fe_to_Al='dilute',
            **other_job_kwargs
        )

    @staticmethod
    def S_ideal_mixing(c, site_fraction=1):
        c = c / site_fraction
        S = -site_fraction * KB * ((1 - c) * np.log(1 - c) + c * np.log(c))
        return S

    def _G_dilute_mixing(self, ideal_energy, defect_concentration, defect_energy, temperature, site_fraction=1):
        return (
                ideal_energy
                + defect_concentration * defect_energy
                - temperature * self.S_ideal_mixing(defect_concentration, site_fraction=site_fraction)
        )

    def plot_G_0K_point_defects(self, temperature=523, c_range=None, potl_index=0, ax=None, beautify=True):
        """
        Assumes no stoichiometry-neutral swaps and that the Fe-rich D03 phase first fills all of the Fe site with the
        lower potential energy cost for swaping to Al, then proceeds to fill the more expensive sites. It's assumed that
        the potential energy cost dominates the extra entropy from one site having double the degeneracy of the other.
        """
        (fig, ax) = plt.subplots() if ax is None else (None, ax)
        c_range = np.linspace(0, 1, 200) if c_range is None else c_range
        eps = 1e-3
        c_range[np.isclose(c_range, 0)] = eps
        c_range[np.isclose(c_range, 1)] = 1 - eps

        bcc = self.get_bcc_peratom_energy(potl_index=potl_index)
        form = self.get_dilute_formation_energy(potl_index=potl_index)[0]
        G_BCC = self._G_dilute_mixing(bcc, c_range, form, temperature)

        d03 = self.get_d03_peratom_energy(potl_index=potl_index)
        d03_Al_to_Fe = self.get_dilute_d03_Al_to_Fe_energy(potl_index=potl_index)[0]
        d03_aFe_to_Al = self.get_dilute_d03_aFe_to_Al_energy(potl_index=potl_index)[0]
        d03_bFe_to_Al = self.get_dilute_d03_bFe_to_Al_energy(potl_index=potl_index)[0]
        sf = self.project.create.structure.FeAl.d03_fractions
        G_D03_low_Al = self._G_dilute_mixing(d03, (0.25 - c_range), d03_Al_to_Fe, temperature, site_fraction=sf.Al)
        if d03_bFe_to_Al < d03_aFe_to_Al:
            form_hi, frac_hi, form_vhi, frac_vhi = d03_bFe_to_Al, sf.bFe, d03_aFe_to_Al, sf.aFe
        else:
            form_hi, frac_hi, form_vhi, frac_vhi = d03_aFe_to_Al, sf.aFe, d03_bFe_to_Al, sf.bFe
        G_D03_hi_Al = self._G_dilute_mixing(d03, (c_range - 0.25), form_hi, temperature, site_fraction=frac_hi)
        G_D03_very_hi_Al = self._G_dilute_mixing(
            d03 + frac_hi * form_hi, (c_range - (0.25 + frac_hi)), form_vhi, temperature, site_fraction=frac_vhi
        )
        G_D03 = (
                np.nan_to_num(G_D03_low_Al, nan=0)
                + np.nan_to_num(G_D03_hi_Al, nan=0)
                + np.nan_to_num(G_D03_very_hi_Al, nan=0)
        )

        b2 = self.get_b2_peratom_energy(potl_index=potl_index)
        b2_Al_to_Fe = self.get_dilute_b2_Al_to_Fe_energy(potl_index=potl_index)[0]
        b2_Fe_to_Al = self.get_dilute_b2_Fe_to_Al_energy(potl_index=potl_index)[0]
        G_B2_low_Al = self._G_dilute_mixing(b2, (0.5 - c_range), b2_Al_to_Fe, temperature, site_fraction=0.5)
        G_B2_hi_Al = self._G_dilute_mixing(b2, (c_range - 0.5), b2_Fe_to_Al, temperature, site_fraction=0.5)
        G_B2 = np.nan_to_num(G_B2_low_Al, nan=0) + np.nan_to_num(G_B2_hi_Al, nan=0)

        ax.plot(c_range, G_BCC, label='BCC')
        ax.plot(c_range, G_D03, label='D03')
        ax.plot(c_range, G_B2, label='B2')
        if beautify:
            ax.set_xlabel('$c_\mathrm{Al}$')
            ax.set_ylabel('$G_\mathrm{phase}$ [eV]')
            fig.legend()
        return ax

    @lru_cache()
    def get_solid_solution_repeats(self, potl_index=0, stderr=1e-3, run_again=False, n_trials=10, c_Al=None):
        c_Al = self.project.input.experimental_data['c_Al'] if c_Al is None else c_Al

        reps = 0
        actual_err = np.inf
        while actual_err > stderr:
            reps += 1

            n_atoms = len(self.project.create.structure.FeAl.bcc(repeat=reps))
            if int(n_atoms * c_Al) < 1:
                continue

            energies = []
            for n in range(n_trials):
                job = self.project.create.job.minimize.bcc(
                    potl_index=potl_index,
                    repeat=reps,
                    c_Al=c_Al,
                    trial=n,
                    delete_existing_job=run_again
                )
                job.run()
                energies.append(job.output.energy_pot[-1] / len(job.structure))
            actual_err = np.std(energies) / np.sqrt(n_trials)
        return reps, actual_err, n_trials

    @lru_cache()
    def get_formation_energies(
            self, c_Al_max=0.25, repeat=1, potl_index=0, stderr=1e-3, run_again=False, n_trials=10
    ):
        converged_reps, _, _ = self.get_solid_solution_repeats(
            potl_index=potl_index, stderr=stderr, run_again=run_again, n_trials=n_trials
        )
        repeat = max(converged_reps, repeat)

        n_atoms = len(self.project.create.structure.FeAl.bcc(repeat=repeat))
        c_Al = (np.arange(n_atoms) + 1) / n_atoms
        c_Al = c_Al[c_Al <= c_Al_max]

        energies = np.nan * np.ones((len(c_Al), n_trials))
        for i, c in enumerate(c_Al):
            for n in np.arange(n_trials):
                job = self.project.create.job.minimize.bcc(
                    potl_index=potl_index,
                    repeat=repeat,
                    c_Al=c,
                    trial=n,
                    delete_existing_job=run_again
                )
                job.run()
                energies[i, n] = job.output.energy_pot[-1] / len(job.structure)
        return c_Al, energies

    @lru_cache()
    def get_d03_Al_to_Fe_energies(
            self, c_antisite_max=0.25, repeat=1, potl_index=0, stderr=1e-3, run_again=False, n_trials=10
    ):
        converged_reps, _, _ = self.get_solid_solution_repeats(
            potl_index=potl_index, stderr=stderr, run_again=run_again, n_trials=n_trials
        )
        repeat = int(max(converged_reps, repeat))

        n_antisites = len(self.project.create.structure.FeAl.d03(repeat=repeat)) \
                      * self.project.create.structure.FeAl.d03_fractions.Al
        c_antisites = (np.arange(n_antisites) + 1) / n_antisites
        c_antisites = c_antisites[c_antisites <= c_antisite_max]

        energies = np.nan * np.ones((len(c_antisites), n_trials))
        for i, c in enumerate(c_antisites):
            for n in np.arange(n_trials):
                job = self.project.create.job.minimize.d03(
                    potl_index=potl_index,
                    repeat=repeat,
                    c_D03_anti_Al_to_Fe=c,
                    trial=n,
                    delete_existing_job=run_again
                )
                job.run()
                energies[i, n] = job.output.energy_pot[-1] / len(job.structure)
        return c_antisites, energies

    # def get_dmu_0K(self, c_Al=0.18, potl_index=0, run_again=False, **other_job_kwargs):
    #     E_BCC = self.get_BCC_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs)
    #     E_D03 = self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs)
    #     E_B2 = self.get_B2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs)
    #     delta_c = 0.25
    #     if c_Al <= 0.25:
    #         return (E_D03 - E_BCC) / delta_c
    #     elif c_Al <= 0.5:
    #         return (E_B2 - E_D03) / delta_c
    #     else:
    #         raise ValueError(f"0K chemical potential only defined for Al concentrations <= 0.5, but got {c_Al}")
