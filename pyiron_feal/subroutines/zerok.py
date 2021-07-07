# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_feal.utils import HasProject
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


class ZeroK(HasProject):

    @staticmethod
    def _get_energy_and_size(job_creator, potl_index=0, run_again=False, **other_job_kwargs):
        job = job_creator(potl_index=potl_index, delete_existing_job=run_again, **other_job_kwargs)
        if isinstance(job, GenericJob):
            job.run()
            return job.output.energy_pot[-1], len(job.structure)
        else:
            return job['output/generic/energy_pot'][-1], len(job['output/structure/indices'])

    def _get_peratom_energy(self, job_creator, potl_index=0, run_again=False, **other_job_kwargs):
        e, n = self._get_energy_and_size(job_creator, potl_index=potl_index, run_again=run_again, **other_job_kwargs)
        return e / n

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
            self.project.create.job.minimize.d03,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    def plot_phases_0K(self, potl_index=0, ax=None, label_points=True, label_x=True, label_y=True, relative=True):
        (fig, ax) = plt.subplots() if ax is None else (None, ax)
        e_fcc = self.get_fcc_peratom_energy(potl_index=potl_index)
        e_bcc = self.get_bcc_peratom_energy(potl_index=potl_index)
        e_d03 = self.get_d03_peratom_energy(potl_index=potl_index)
        e_b2 = self.get_b2_peratom_energy(potl_index=potl_index)

        c_Al = [0, 0, 0.25, 0.5]
        energies = np.array([e_fcc, e_bcc, e_d03, e_b2])
        if relative:
            energies -= e_bcc
        ax.plot(c_Al, energies, marker='o')
        if label_points:
            for c, E, label in zip(c_Al, energies, ['FCC', 'BCC', '$\mathrm{D0}_3$', 'B2']):
                ax.annotate(label, (c, E))
        if label_x:
            ax.set_xlabel('$c_\mathrm{Al}$')
        if label_y:
            ax.set_ylabel('$E$ [eV/atom]')

        return ax

    def _get_size_converged_point_defect_energy(
            self, ref_energy_per_atom, job_creator, stderr=1e-3, potl_index=0, run_again=False, **other_job_kwargs
    ):
        reps = 0
        actual_err = np.inf
        energies = [np.inf]
        while actual_err > stderr:
            reps += 1
            e, n = self._get_energy_and_size(
                job_creator,
                potl_index=potl_index,
                run_again=run_again,
                **other_job_kwargs
            )
            energies.append(e - n * ref_energy_per_atom)
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

    def dilute_point_defect_dictionary(self, potl_index=0):
        return {
            'BCC': {
                'formation': self.get_dilute_formation_energy(potl_index=potl_index)
            },
            'D03': {
                'Al->Fe': self.get_dilute_d03_Al_to_Fe_energy(potl_index=potl_index),
                'aFe->Al': self.get_dilute_d03_aFe_to_Al_energy(potl_index=potl_index),
                'bFe->Al': self.get_dilute_d03_bFe_to_Al_energy(potl_index=potl_index)
            },
            'B2': {
                'Al->Fe': self.get_dilute_b2_Al_to_Fe_energy(potl_index=potl_index),
                'Fe->Al': self.get_dilute_b2_Fe_to_Al_energy(potl_index=potl_index)
            }
        }

    @staticmethod
    def S_ideal_mixing(c, site_fraction=1):
        c = c / site_fraction
        S = -site_fraction * KB * ((1 - c) * np.log(1 - c) + c * np.log(c))
        return S

    def G_dilute_mixing(self, ideal_energy, defect_concentration, defect_energy, temperature, site_fraction=1):
        return (
                ideal_energy
                + defect_concentration * defect_energy
                - temperature * self.S_ideal_mixing(defect_concentration, site_fraction=site_fraction)
        )

    def G_mixing(self, potential_energy, defect_concentration, temperature, site_fraction=1):
        return (
                potential_energy
                - temperature * self.S_ideal_mixing(defect_concentration, site_fraction=site_fraction)
        )

    def plot_G_0K_point_defects(
            self, temperature=523, c_range=None, potl_index=0, ax=None, label_x=True, label_y=True, legend=True):
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
        G_BCC = self.G_dilute_mixing(bcc, c_range, form, temperature)

        d03 = self.get_d03_peratom_energy(potl_index=potl_index)
        d03_Al_to_Fe = self.get_dilute_d03_Al_to_Fe_energy(potl_index=potl_index)[0]
        d03_aFe_to_Al = self.get_dilute_d03_aFe_to_Al_energy(potl_index=potl_index)[0]
        d03_bFe_to_Al = self.get_dilute_d03_bFe_to_Al_energy(potl_index=potl_index)[0]
        sf = self.project.create.structure.FeAl.d03_fractions
        G_D03_low_Al = self.G_dilute_mixing(d03, (0.25 - c_range), d03_Al_to_Fe, temperature, site_fraction=sf.Al)
        if d03_bFe_to_Al < d03_aFe_to_Al:
            form_hi, frac_hi, form_vhi, frac_vhi = d03_bFe_to_Al, sf.bFe, d03_aFe_to_Al, sf.aFe
        else:
            form_hi, frac_hi, form_vhi, frac_vhi = d03_aFe_to_Al, sf.aFe, d03_bFe_to_Al, sf.bFe
        G_D03_hi_Al = self.G_dilute_mixing(d03, (c_range - 0.25), form_hi, temperature, site_fraction=frac_hi)
        G_D03_very_hi_Al = self.G_dilute_mixing(
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
        G_B2_low_Al = self.G_dilute_mixing(b2, (0.5 - c_range), b2_Al_to_Fe, temperature, site_fraction=0.5)
        G_B2_hi_Al = self.G_dilute_mixing(b2, (c_range - 0.5), b2_Fe_to_Al, temperature, site_fraction=0.5)
        G_B2 = np.nan_to_num(G_B2_low_Al, nan=0) + np.nan_to_num(G_B2_hi_Al, nan=0)

        ax.plot(c_range, G_BCC, label='BCC')
        ax.plot(c_range, G_D03, label='D03', linestyle='--')
        ax.plot(c_range, G_B2, label='B2', linestyle=':')
        if label_x:
            ax.set_xlabel('$c_\mathrm{Al}$')
        if label_y:
            ax.set_ylabel('$G_\mathrm{phase}$ [eV]')
        if legend:
            ax.legend()
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
                energies.append(
                    self._get_peratom_energy(
                        self.project.create.job.minimize.bcc,
                        potl_index=potl_index,
                        run_again=run_again,
                        repeat=reps,
                        c_Al=c_Al,
                        trial=n,
                    )
                )
            actual_err = np.std(energies) / np.sqrt(n_trials)
        return reps, actual_err, n_trials

    def _at_least_converged_repetions(self, repeat, potl_index, stderr, run_again, n_trials):
        converged_reps, _, _ = self.get_solid_solution_repeats(
            potl_index=potl_index, stderr=stderr, run_again=run_again, n_trials=n_trials
        )
        return max(converged_reps, repeat)

    @staticmethod
    def _most_similar_range(desired_concentrations, n_sites):
        allowable_concentrations = np.arange(n_sites + 1) / n_sites
        return allowable_concentrations[
            np.unique([np.argmin(np.abs(allowable_concentrations - c)) for c in desired_concentrations])
        ]

    def get_bcc_solution_energies(
            self,
            concentrations,
            repeat=1,
            potl_index=0,
            stderr=1e-3,
            run_again=False,
            n_trials=10,
    ):
        repeat = self._at_least_converged_repetions(repeat, potl_index, stderr, run_again, n_trials)
        c_Al = self._most_similar_range(
            concentrations,
            len(self.project.create.structure.FeAl.bcc(repeat=repeat))
        )

        energies = np.nan * np.ones((len(c_Al), n_trials))
        for i, c in enumerate(c_Al):
            for n in np.arange(n_trials):
                energies[i, n] = self._get_peratom_energy(
                    self.project.create.job.minimize.bcc,
                    potl_index=potl_index,
                    run_again=run_again,
                    repeat=repeat,
                    c_Al=c,
                    trial=n
                )
        return c_Al, energies

    def get_d03_Al_to_Fe_energies(
            self,
            concentrations,
            repeat=1,
            potl_index=0,
            stderr=1e-3,
            run_again=False,
            n_trials=10,
    ):
        repeat = self._at_least_converged_repetions(repeat, potl_index, stderr, run_again, n_trials)
        n_sites = len(self.project.create.structure.FeAl.d03(repeat=repeat))
        n_antisites = n_sites * self.project.create.structure.FeAl.d03_fractions.Al
        c_antisites = self._most_similar_range(concentrations, n_antisites)

        energies = np.nan * np.ones((len(c_antisites), n_trials))
        for i, c in enumerate(c_antisites):
            for n in np.arange(n_trials):
                energies[i, n] = self._get_peratom_energy(
                    self.project.create.job.minimize.d03,
                    run_again=run_again,
                    potl_index=potl_index,
                    repeat=repeat,
                    c_D03_anti_Al_to_Fe=c,
                    trial=n,
                )
        return c_antisites, energies

    def get_interactive_cluster_data(
            self,
            reference_structure_creator,
            potl_index=0,
            a=None,
            repeat=3,
            trial=0,
            pressure=0,
            run_again=False,
            c_Al=None,
            cluster_cell_fraction_max=1/8
    ):
        job = self.project.create.job.minimize.interactive_cluster(
            potl_index=potl_index,
            a=a,
            repeat=repeat,
            trial=trial,
            pressure=pressure,
            delete_existing_job=run_again,
            c_Al=c_Al,
            max_cluster_fraction=0.125,
            symbol_ref=reference_structure_creator.__name__
        )
        if job.status != 'finished':
            ref_symbols = reference_structure_creator(a=a, repeat=repeat).get_chemical_symbols()
            neigh = job.structure.get_neighbors(num_neighbors=8)
            cluster = []
            concentration = []
            cluster_volume = []
            job.interactive_open()
            for _ in np.arange(int(len(job.structure) * cluster_cell_fraction_max)):
                all_neighbors = neigh.indices[cluster].flatten() if len(cluster) > 0 else np.array([0])
                possible_new_neighbors = all_neighbors[~np.in1d(all_neighbors, cluster)]
                i = np.random.choice(possible_new_neighbors, 1)[0]
                job.structure[i] = ref_symbols[i]
                cluster.append(i)
                job.run()
                relaxed_structure = job.get_structure()
                concentration.append(np.mean(relaxed_structure.get_chemical_symbols() == 'Al'))
                cluster_volume.append(np.sum(relaxed_structure.analyse.pyscal_voronoi_volume()[cluster]))
                job.interactive_structure_setter(relaxed_structure)
            job.interactive_close()
            with job._hdf5.open('output/special') as hdf:
                hdf['cluster'] = cluster
                hdf['concentration'] = concentration
                hdf['clustervolume'] = cluster_volume

        return (
            job,
            job['output/special/cluster'],
            job['output/special/concentration'],
            job['output/special/clustervolume'],
            job['output/generic/energy_pot']
        )