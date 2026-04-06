#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:19:32 2025

@author: Rich
"""

import numpy as np
from math import exp, sqrt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

#---constants---
arad = 7.5657e-15
c = 3.0e10
k_B = 1.38e-16
m_p = 1.67e-24
m_e = 9.10938356e-28
m_u = 1.660539e-24
hbar = 6.62607015e-27/(2*np.pi)
e_cgs = 4.80320425e-10
G = 6.6743e-8

#---NS parameters---
g = 1.90e14
mdot_Edd = 8.8e4

X_h = 0.70
X_he = 0.30
Ye = 1.0 - X_he/2.0
Yi = 1.0 - 3.0*X_he/4.0

#---nuclear Q factors---
MeV_to_erg_per_nucleon = 1.602e-6
Q3alpha = 7.275 * MeV_to_erg_per_nucleon / m_p
Qbase = 0.15 * MeV_to_erg_per_nucleon / m_p

#---https://github.com/andrewcumming/onezone + paczynski---
def pressure(rho, T):
    # degenerate + thermal electrons
    P_deg = 9.91e12 * (Ye * rho)**(5.0/3.0)
    P_therm_e = Ye * rho * k_B * T / m_p
    P_e = sqrt(P_deg**2 + P_therm_e**2)
    # ion ideal gas
    P_i = Yi * rho * k_B * T / m_p
    # radiation
    P_rad = arad * T**4 / 3.0
    return P_e + P_i + P_rad 
    #return P_deg

def find_rho_eqn(rho,P,T):
    return pressure(rho,T)-P
    
def find_density(P, T):
    '''
    do a numerical inversion to find density
    '''
    rho = brentq(find_rho_eqn,1.0,1e12,xtol=1e-6,args=(P,T))
    return rho

#---nuclear burnig and cooling rates---
def epsilon_3alpha(T, rho, X_he):
    T8 = T*1e-8
    rho5 = rho*1e-5
    eps = 5.3e21*(rho5**2)*(X_he**3)*(T8**-3)*exp(-44.0/T8)
    # H/He burning into Fe
    eps *= (1.6 + (1.0 - X_he)*4.9)/0.606
    return eps

def epsilon_total(T, y, rho, X_he, mdot_frac):
    eps = epsilon_3alpha(T, rho, X_he)
    #eps = 0 # remove this to bring back 3alpha
    eps += 5.8e15*0.01 # CNO
    eps += Qbase * (mdot_frac*mdot_Edd)/y # heating from crust
    return eps

def epsilon_cool(T, y, rho):
    # this is actaully the F/y term, but it is equivalent as an epsilon
    kappa = kappa_total(T,rho)
    return c*arad*T**4/(3*kappa*y**2)

#---opacities---
def kappa_es(T, rho, X_H):
    term_deg = 1.0 + 2.7e11*(rho/(T**2))
    term_kn = 1.0 + (T/4.5e8)**0.86
    return 0.2*(1.0 + X_H)/(term_deg*term_kn)

def kappa_ff(T, rho, Ye_local, X_he_local):
    mu_e = 1.0/Ye_local
    T8 = T*1e-8
    rho5 = rho*1e-5
    YHe = X_he_local/4.0
    YC = (1.0 - X_he_local)/12.0
    sumZ = 4.0*YHe + 36.0*YC
    return 0.753*rho5*(T8**-3.5)*mu_e*sumZ

def kappa_cond(rho, T, Ye_local, Yi_local, X_he_local):
    n_e = Ye_local*rho/m_u
    n_i = Yi_local*rho/m_u
    p_F = hbar*(3*np.pi**2*n_e)**(1.0/3.0)
    E_F = sqrt((p_F*c)**2 + (m_e*c**2)**2) - m_e*c**2
    m_s = m_e + E_F/c**2
    YHe = X_he_local/4.0
    YC  = (1.0 - X_he_local)/12.0
    sumZ = 4.0*YHe + 36.0*YC
    nu_ei = (4*e_cgs**4*m_s)/(3*np.pi*hbar**3)*(sumZ/Ye_local)
    Kcond = ( (n_e**2)*k_B**2*T )/(n_i*m_s*nu_ei)
    return (4*arad*c*T**3)/(3*rho*Kcond)

def kappa_total(T,rho):
    k_es = kappa_es(T,rho,X_h)
    k_ff = kappa_ff(T,rho,Ye,X_he)
    k_r = k_es + k_ff
    k_c = kappa_cond(rho,T,Ye,Yi,X_he)

    return 1.0/(1.0/k_r + 1.0/k_c)
    return 0.2

#---heat capacity---
def cp_total(T,rho):
    # Ion + electron + radiation
    c_p = 2.5*(Yi + Ye)*k_B/m_p
    T8 = T*1e-8
    rho5 = rho*1e-5
    c_p += 4.0*arad*T8**3*1e19/rho5
    return c_p

#---construct ODE and solve---
def derivs(t, state, mdot_frac):
    T, y = state
    rho = find_density(g*y, T)

    eps_nuc = epsilon_total(T,y,rho,X_he,mdot_frac)
    eps_3 = epsilon_3alpha(T,rho,X_he)
    eps_loss = epsilon_cool(T,y,rho)

    return [(eps_nuc - eps_loss)/cp_total(T,rho),
        (mdot_frac*mdot_Edd) - 12*eps_3*y/Q3alpha]

#---compute flux after having T,y solved---
def compute_flux(T_arr,y_arr):
    F = np.zeros_like(T_arr)
    for i,(T,y) in enumerate(zip(T_arr,y_arr)):
        rho = find_density(g*y,T)
        F[i]=c*arad*T**4/(3*kappa_total(T,rho)*y)
    return F


#---main code---
mdot_list = [1, 2.0, 3.0]
labels    = [f"{m} $\\dot{{m}}_{{\\rm Edd}}$" for m in mdot_list]

# Figure 1: Flux vs time (3 panels)
fig1, axes1 = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
# Figure 2: Phase diagram T vs y (3 panels)
fig2, axes2 = plt.subplots(3, 1, figsize=(5, 8), sharex=True)

for i, (md, label) in enumerate(zip(mdot_list, labels)):
    sol  = solve_ivp(derivs, [0, 60*50], [2e8, 2e8], args=(md,),
                     method="BDF", rtol=1e-8, atol=1e-10)  # stiff solver
    tmin = sol.t / 60.0
    T = sol.y[0]
    y = sol.y[1]
    F = compute_flux(T, y)
    
    #---F vs t---
    ax1 = axes1[i]
    ax1.plot(tmin, F, 'C1')
    ax1.set_yscale('log')
    ax1.set_ylim(1e21, 1e25)
    ax1.set_title(label, fontsize = 15)
    ax1.set_ylabel("$F$ (erg cm$^{-2}$ s$^{-1}$)", fontsize = 15)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    #---T vs y---
    ax2 = axes2[i]
    ax2.plot(y*1e-8, T*1e-9, lw=1.6, color = 'C0')
    ax2.set_ylabel(r"$T_9~(\mathrm{K})$", fontsize = 15)
    ax2.set_title(label, fontsize = 15)
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    

axes1[-1].set_xlabel("$t$ (min)", fontsize = 15)
#axes1[0].set_xlim(0, 50)
fig1.tight_layout()
fig1.savefig("onezone_flux_simple.pdf", bbox_inches='tight')


axes2[-1].set_xlabel(r"$y_8$ (g cm$^{-2}$)", fontsize = 15)
fig2.tight_layout()
fig2.savefig("onezone_phase_simple.pdf", bbox_inches='tight')

plt.show()