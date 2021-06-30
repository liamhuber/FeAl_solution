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
    def get_BCC_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.BCC,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_FCC_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.FCC,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_B2_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.B2,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_peratom_energy(self, potl_index=0, run_again=False, **other_job_kwargs):
        return self._get_peratom_energy(
            self.project.create.job.minimize.D03,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

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
    def get_formation_energy(self, stderr=1e-3, run_again=False, potl_index=0, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_BCC_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.dilute,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_B2_antisite_energy_Fe_to_Al(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_B2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.B2_anti_Fe_to_Al,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_B2_antisite_energy_Al_to_Fe(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_B2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.B2_anti_Al_to_Fe,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_antisite_energy_Al_to_Fe(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.D03_anti_Al_to_Fe,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_antisite_energy_aFe_to_Al(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.D03_anti_aFe_to_Al,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_D03_antisite_energy_bFe_to_Al(self, potl_index=0, stderr=1e-3, run_again=False, **other_job_kwargs):
        return self._get_size_converged_point_defect_energy(
            self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs),
            self.project.create.job.minimize.D03_anti_bFe_to_Al,
            stderr=stderr,
            potl_index=potl_index,
            run_again=run_again,
            **other_job_kwargs
        )

    @lru_cache()
    def get_solid_solution_repeats(self, potl_index=0, stderr=1e-3, run_again=False, n_trials=10, c_Al=None):
        c_Al = self.project.input.experimental_data['c_Al'] if c_Al is None else c_Al
        reps = 0
        actual_err = np.inf
        while actual_err > stderr:
            reps += 1
            if int(2 * reps ** 3 * c_Al) < 1:
                continue
            energies = []
            for n in range(n_trials):
                job = self.project.create.job.minimize.random_BCC(
                    potl_index=potl_index,
                    repeat=reps,
                    c_Al=c_Al,
                    trial=n,
                    delete_existing_job=run_again
                )
                job.run()
                energies.append(job.output.energy_pot[-1] / len(job.structure))
            actual_err = np.std(energies) / np.sqrt(n_trials)
            print(f"stderr = {actual_err}")
        return reps, actual_err

    def plot_phases_0K(self, potl_index=0, ax=None, beautify=True):
        (fig, ax) = plt.subplots() if ax is None else (None, ax)
        E_FCC = self.get_FCC_peratom_energy(potl_index=potl_index)
        E_BCC = self.get_BCC_peratom_energy(potl_index=potl_index)
        E_D03 = self.get_D03_peratom_energy(potl_index=potl_index)
        E_B2 = self.get_B2_peratom_energy(potl_index=potl_index)

        c_Al = [0, 0, 0.25, 0.5]
        energies = [E_FCC, E_BCC, E_D03, E_B2]
        ax.plot(c_Al, energies, marker='o')
        if beautify:
            ax.set_xlabel('$c_\mathrm{Al}$')
            ax.set_ylabel('$E$ [eV/atom]')
            for c, E, label in zip(c_Al, energies, ['FCC', 'BCC', 'D03', 'B2']):
                ax.annotate(label, (c, E))
        return ax

    def get_dmu_0K(self, c_Al=0.18, potl_index=0, run_again=False, **other_job_kwargs):
        E_BCC = self.get_BCC_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs)
        E_D03 = self.get_D03_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs)
        E_B2 = self.get_B2_peratom_energy(potl_index=potl_index, run_again=run_again, **other_job_kwargs)
        delta_c = 0.25
        if c_Al <= 0.25:
            return (E_D03 - E_BCC) / delta_c
        elif c_Al <= 0.5:
            return (E_B2 - E_D03) / delta_c
        else:
            raise ValueError(f"0K chemical potential only defined for Al concentrations <= 0.5, but got {c_Al}")

