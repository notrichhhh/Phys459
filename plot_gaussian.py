#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
mu = np.pi / 2
sigma = 0.5
mdot0 = 1.0

# -----------------------------
# Theta grid
# -----------------------------
N_theta = 500
theta = np.linspace(0, np.pi, N_theta)

# -----------------------------
# Gaussian profile
# -----------------------------
gaussian_profile = np.exp(-((theta - mu)**2) / (2 * sigma**2))
mdot_frac = mdot0 * gaussian_profile

# -----------------------------
# Plot (MATCHED STYLE)
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot(theta, mdot_frac, color='C0')

# Labels
ax.set_xlabel(r'$\theta$ (rad)', fontsize=15)
ax.set_ylabel(r'$\dot{m}_{\mathrm{frac}}$', fontsize=15)

# Title
ax.set_title('a) Gaussian Accretion Profile vs Theta', fontsize=15)

# Equator line
ax.axvline(mu, linestyle='--', linewidth=1, label=r'Equator ($\pi/2$)')

# Grid (match your code)
ax.grid(alpha=0.3)

# Tick sizes (match your code)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

# Legend
ax.legend(fontsize=12)

# Layout + save (match your workflow)
fig.tight_layout()
fig.savefig('Gaussian_accretion_profile.pdf', bbox_inches='tight')

plt.show()