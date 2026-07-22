"""Figures for the legacy ideal prescribed-coupling benchmark."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .constants import Gamma0, kB, T_K
from .theory import legacy_K_WKB_proxy, F_bound, tau_gate, K_star, K_demo
from .grid import z_um, z_np, to_np
from .gpe import make_potential, integ
from .universality import bell, bell_F, CNOT_mat

# Matplotlib style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.8,
    'lines.linewidth': 1.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
C1 = '#1f77b4'; C2 = '#d62728'; C3 = '#2ca02c'; CB = '#ff7f0e'


def make_fig1(save=True):
    """Figure 1: Analytical theory — WKB coupling and fidelity bound."""
    fig, ax1 = plt.subplots(2, 2, figsize=(7, 5.5))
    fig.subplots_adjust(hspace=0.44, wspace=0.38)

    d_arr  = np.linspace(0.3, 4.0, 300) * 1e-6
    dB_arr = np.linspace(0.5, 50.0, 300) * 1e-3
    K_arr  = np.logspace(6.5, 10, 400)

    ax = ax1[0, 0]
    ax.semilogy(d_arr * 1e6, legacy_K_WKB_proxy(d_arr, 5e-3) / (2 * np.pi * 1e6), 'k', lw=1.5)
    ax.set_xlabel(r'$d$ (µm)'); ax.set_ylabel(r'$K_{\max}/2\pi$ (MHz)')
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.text(0.97, 0.92, r'$\delta B=5\,$mT', transform=ax.transAxes, ha='right', fontsize=8)
    ax.grid(True, which='both', alpha=0.2, ls=':', lw=0.5)

    ax = ax1[0, 1]
    ax.semilogy(dB_arr * 1e3, legacy_K_WKB_proxy(1e-6, dB_arr) / (2 * np.pi * 1e6), 'k', lw=1.5)
    ax.set_xlabel(r'$\delta B$ (mT)'); ax.set_ylabel(r'$K_{\max}/2\pi$ (MHz)')
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.text(0.97, 0.92, r'$d=1\,\mu$m', transform=ax.transAxes, ha='right', fontsize=8)
    ax.grid(True, which='both', alpha=0.2, ls=':', lw=0.5)

    ax = ax1[1, 0]
    ax.semilogx(K_arr / (2 * np.pi * 1e6), F_bound(K_arr), 'k', lw=1.5,
                label=r'$F_{\max}=1-\pi\Gamma/K$')
    ax.axhline(0.99, color='red', ls='--', lw=1.0, label='$F=0.99$')
    ax.axvline(K_star / (2 * np.pi * 1e6), color=C3, ls=':', lw=1.0,
               label=f'$K^*={K_star/(2*np.pi*1e6):.0f}$ MHz')
    ax.set_xlabel(r'$K_{\max}/2\pi$ (MHz)'); ax.set_ylabel(r'$F_{\max}$')
    ax.set_title('(c)', loc='left', fontweight='bold'); ax.set_ylim(0.60, 1.02)
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(True, which='both', alpha=0.2, ls=':', lw=0.5)

    ax = ax1[1, 1]
    V_show = to_np(make_potential())
    ax.plot(z_um, V_show / (2 * np.pi * 1e9), 'k', lw=1.5)
    ax.fill_between(z_um, V_show / (2 * np.pi * 1e9), 0,
                    where=np.abs(z_np) < 0.5e-6, color=CB, alpha=0.5, label='barrier')
    ax.set_xlabel(r'$z$ (µm)'); ax.set_ylabel(r'$V/2\pi$ (GHz)')
    ax.set_title('(d)', loc='left', fontweight='bold')
    ax.legend(loc='upper center', fontsize=8); ax.set_xlim(-22, 22)
    ax.grid(True, alpha=0.2, ls=':', lw=0.5)

    fig.suptitle('Legacy carrier-prefactor proxy and ideal loss benchmark',
                 fontsize=10, fontweight='bold')
    if save:
        fig.savefig('fig1_theory.pdf'); fig.savefig('fig1_theory.png', dpi=200)
    return fig


def make_fig2(sims, tt_F, save=True):
    """Figure 2: GPE dynamics — truth table, density map, Bloch sphere."""
    truth_cases = [
        ('0', '0', r'$|00\rangle\!\to\!|00\rangle$', '0'),
        ('0', '1', r'$|01\rangle\!\to\!|01\rangle$', '1'),
        ('1', '0', r'$|10\rangle\!\to\!|11\rangle$', '1'),
        ('1', '1', r'$|11\rangle\!\to\!|10\rangle$', '0'),
    ]
    fig = plt.figure(figsize=(7, 8.5))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.56, wspace=0.42)
    panel_lbl = ['(a)', '(b)', '(c)', '(d)']

    for k, (ctrl, tgt, lbl, eB) in enumerate(truth_cases):
        ax = fig.add_subplot(gs[0, k])
        s = sims[(ctrl, tgt)]
        nBp, nBm = integ(s, 'B'); _, nAm = integ(s, 'A')
        ax.plot(s['t_ns'], nBp, C1, lw=1.5, label=r'$N_{B+}$')
        ax.plot(s['t_ns'], nBm, C2, lw=1.5, label=r'$N_{B-}$')
        ax.plot(s['t_ns'], nAm, 'k--', lw=0.8, alpha=0.7, label=r'$N_{A-}$')
        ax.set_xlabel('$t$ (ns)', labelpad=1); ax.set_ylabel('population', labelpad=1)
        ax.set_title(f'{panel_lbl[k]} {lbl}', fontsize=8, pad=2)
        ax.set_ylim(-0.05, 1.12)
        if k == 0: ax.legend(loc='center right', fontsize=6.5, framealpha=0.7)
        ax.grid(True, alpha=0.2, ls=':', lw=0.5)

    # (e) density map
    ax_d = fig.add_subplot(gs[1, :])
    sf = sims[('1', '0')]
    dens = sf['N_Bp'] + sf['N_Bm'] + sf['N_Ap'] + sf['N_Am']
    ext = [sf['t_ns'][0], sf['t_ns'][-1], sf['z_um'][0], sf['z_um'][-1]]
    im = ax_d.imshow(dens.T, origin='lower', aspect='auto', extent=ext, cmap='magma')
    plt.colorbar(im, ax=ax_d, label=r'$\sum|\psi|^2$', pad=0.01, fraction=0.018)
    ax_d.axhline(0.5, color='cyan', ls='--', lw=0.8, alpha=0.8)
    ax_d.axhline(-0.5, color='cyan', ls='--', lw=0.8, alpha=0.8)
    ax_d.set_xlabel('$t$ (ns)'); ax_d.set_ylabel(r'$z$ (µm)')
    ax_d.set_title(r'(e)  $|\psi(z,t)|^2$ — case $|10\rangle\!\to\!|11\rangle$',
                   loc='left', fontweight='bold', fontsize=9)

    # (f) Bloch sphere
    ax_bs = fig.add_subplot(gs[2, :2], projection='3d')
    nBp_r, nBm_r = integ(sf, 'B')
    th = np.arccos(np.clip(nBp_r - nBm_r, -1, 1))
    ph = sf['phi_B']
    bx = np.sin(th) * np.cos(ph); by = np.sin(th) * np.sin(ph); bz = np.cos(th)
    u = np.linspace(0, 2 * np.pi, 25); v = np.linspace(0, np.pi, 25)
    ax_bs.plot_wireframe(np.outer(np.cos(u), np.sin(v)),
                         np.outer(np.sin(u), np.sin(v)),
                         np.outer(np.ones_like(u), np.cos(v)),
                         color='gray', alpha=0.12, lw=0.3)
    ax_bs.plot(bx, by, bz, C3, lw=1.8)
    ax_bs.scatter([bx[0]], [by[0]], [bz[0]], color=C1, s=35, zorder=5)
    ax_bs.scatter([bx[-1]], [by[-1]], [bz[-1]], color=C2, s=35, zorder=5)
    ax_bs.text(0, 0, 1.25, r'$|0\rangle$', ha='center', fontsize=9, color=C1)
    ax_bs.text(0, 0, -1.32, r'$|1\rangle$', ha='center', fontsize=9, color=C2)
    ax_bs.set_xlim(-1.3, 1.3); ax_bs.set_ylim(-1.3, 1.3); ax_bs.set_zlim(-1.3, 1.3)
    ax_bs.set_xlabel('x', labelpad=-4); ax_bs.set_ylabel('y', labelpad=-4)
    ax_bs.set_zlabel('z', labelpad=-4)
    ax_bs.set_title('(f) Bloch sphere', loc='left', fontweight='bold', fontsize=9)
    ax_bs.tick_params(labelsize=7)

    # (g) truth-table bar
    ax_tt = fig.add_subplot(gs[2, 2:])
    cols = [C1, C1, C2, C2]
    ax_tt.bar(range(4), tt_F, color=cols, edgecolor='k', lw=0.5)
    ax_tt.set_xticks(range(4))
    ax_tt.set_xticklabels([r'$|00\rangle$', r'$|01\rangle$',
                           r'$|10\rangle$', r'$|11\rangle$'])
    ax_tt.axhline(0.99, color='red', ls='--', lw=1.0, label='$F=0.99$')
    ax_tt.set_ylim(0, 1.15); ax_tt.set_ylabel('Fidelity $F$')
    ax_tt.set_title(f'(g) $\\bar{{F}}={np.mean(tt_F):.3f}$ (pumped)',
                    loc='left', fontweight='bold', fontsize=9)
    for k2, v in enumerate(tt_F):
        ax_tt.text(k2, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
    ax_tt.legend(fontsize=8); ax_tt.grid(True, axis='y', alpha=0.2, ls=':', lw=0.5)

    T_demo = tau_gate(K_demo) * 1e9
    fig.suptitle('GPE truth-table simulation  '
                 rf'($K_{{\max}}/2\pi=250\,$MHz, $T_{{\rm gate}}={T_demo:.1f}\,$ns)',
                 fontsize=10, fontweight='bold')
    if save:
        fig.savefig('fig2_dynamics.pdf'); fig.savefig('fig2_dynamics.png', dpi=200)
    return fig


def make_fig3(F_KG, K_sw, G_sw, F_Kd, d_sw, dB_sw, K_Kd, idx, save=True):
    """Figure 3: Legacy parameter sweeps and ideal matrix-algebra reference."""
    fig, ax3 = plt.subplots(1, 3, figsize=(10, 3.8))
    fig.subplots_adjust(wspace=0.40)

    # (a) F(K, Gamma)
    ax = ax3[0]
    im = ax.pcolormesh(G_sw / (2 * np.pi * 1e6), K_sw / (2 * np.pi * 1e6), F_KG,
                       shading='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
    cs = ax.contour(G_sw / (2 * np.pi * 1e6), K_sw / (2 * np.pi * 1e6), F_KG,
                    levels=[0.90, 0.95, 0.99], colors='k', linewidths=0.8)
    ax.clabel(cs, fmt='%.2f', fontsize=7)
    ax.axhline(K_demo / (2 * np.pi * 1e6), color='red', ls='--', lw=1.0)
    ax.axvline(Gamma0 / (2 * np.pi * 1e6), color='red', ls='--', lw=1.0,
               label=r'YIG $\alpha{=}2{\times}10^{-4}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\Gamma/2\pi$ (MHz)'); ax.set_ylabel(r'$K_{\max}/2\pi$ (MHz)')
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    plt.colorbar(im, ax=ax, label='Fidelity $F$', pad=0.02)

    # (b) F(d, dB)
    ax = ax3[1]
    im = ax.pcolormesh(dB_sw, d_sw, F_Kd, shading='auto',
                       cmap='RdYlGn', vmin=0.5, vmax=1.0)
    cs = ax.contour(dB_sw, d_sw, F_Kd,
                    levels=[0.90, 0.95, 0.99], colors='k', linewidths=0.8)
    ax.clabel(cs, fmt='%.2f', fontsize=7)
    ax.scatter([dB_sw[idx[1]]], [d_sw[idx[0]]],
               marker='*', s=150, color='k', zorder=5,
               label=f'd={d_sw[idx[0]]:.1f}µm\nδB={dB_sw[idx[1]]:.0f}mT')
    ax.set_xlabel(r'$\delta B$ (mT)'); ax.set_ylabel(r'$d$ (µm)')
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    plt.colorbar(im, ax=ax, label='Fidelity $F$', pad=0.02)

    # (c) Formal complex-amplitude reference; not physical entanglement.
    ax = ax3[2]
    amps = np.abs(bell)**2
    ax.bar(range(4), amps, color=[C1, 'lightgray', 'lightgray', C1],
           edgecolor='k', lw=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([r'$|00\rangle$', r'$|01\rangle$',
                        r'$|10\rangle$', r'$|11\rangle$'])
    ax.set_ylim(0, 0.65); ax.set_ylabel('Probability')
    ax.set_title(f'(c) Formal vector overlap  $F={bell_F:.4f}$',
                 loc='left', fontweight='bold', fontsize=9)
    for k2, v in enumerate(amps):
        if v > 0.01: ax.text(k2, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    ax_in = ax.inset_axes([0.20, 0.25, 0.50, 0.60])
    ax_in.imshow(np.abs(CNOT_mat)**2, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    labs = [r'$|00\rangle$', r'$|01\rangle$', r'$|10\rangle$', r'$|11\rangle$']
    ax_in.set_xticks(range(4)); ax_in.set_yticks(range(4))
    ax_in.set_xticklabels(labs, fontsize=5.5)
    ax_in.set_yticklabels(labs, fontsize=5.5)
    ax_in.set_title(r'$|U|^2$', fontsize=7)

    fig.suptitle('Legacy sweeps and ideal algebra reference', fontsize=10, fontweight='bold')
    if save:
        fig.savefig('fig3_sweep.pdf'); fig.savefig('fig3_sweep.png', dpi=200)
    return fig


def make_fig4(F_ens, F_err, F_ens_mean, F_ens_std,
              noise_sw, F_n_sw, K_ens_sw, F_K_sw, save=True):
    """Figure 4: Ensemble fidelity."""
    from .constants import sig_th
    fig, ax4 = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.subplots_adjust(wspace=0.38)

    ax = ax4[0]
    ax.plot(noise_sw, F_n_sw, 'ko-', ms=5, lw=1.5)
    ax.axvline(sig_th, color='red', ls='--', lw=1.0,
               label=f'300 K noise\n$\\sigma_{{th}}={sig_th:.4f}$')
    ax.axhline(0.99, color=C3, ls=':', lw=1.0, label='$F=0.99$')
    ax.set_xlabel(r'Noise amplitude $\sigma$'); ax.set_ylabel('Ensemble $F$')
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.set_ylim(0.75, 1.02); ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2, ls=':', lw=0.5)

    ax = ax4[1]
    ax.semilogx(K_ens_sw / (2 * np.pi * 1e6), F_K_sw, 'ko-', ms=4, lw=1.5,
                label='Ensemble (N=200)')
    Ka = np.logspace(7.5, 9.5, 200)
    ax.semilogx(Ka / (2 * np.pi * 1e6), F_bound(Ka), 'b--', lw=1.2,
                label=r'$1-\pi\Gamma/K$')
    ax.axvline(K_demo / (2 * np.pi * 1e6), color='red', ls='--', lw=1.0, label='Op. pt.')
    ax.axhline(0.99, color=C3, ls=':', lw=1.0)
    ax.set_xlabel(r'$K_{\max}/2\pi$ (MHz)'); ax.set_ylabel('Fidelity $F$')
    ax.set_title('(b)', loc='left', fontweight='bold')
    ax.set_ylim(0.5, 1.02); ax.legend(fontsize=7.0)
    ax.grid(True, which='both', alpha=0.2, ls=':', lw=0.5)

    ax = ax4[2]
    ax.bar(range(4), F_ens, color=[C1, C1, C2, C2],
           yerr=F_err, capsize=4, edgecolor='k', lw=0.5, error_kw={'lw': 1.0})
    ax.set_xticks(range(4))
    ax.set_xticklabels([r'$|00\rangle$', r'$|01\rangle$',
                        r'$|10\rangle$', r'$|11\rangle$'])
    ax.axhline(0.99, color='red', ls='--', lw=1.0, label='$F=0.99$')
    ax.set_ylim(0.80, 1.06); ax.set_ylabel('Fidelity $F$')
    ax.set_title(f'(c) $\\bar{{F}}={F_ens_mean:.4f}\\pm{F_ens_std:.4f}$',
                 loc='left', fontweight='bold', fontsize=9)
    for k2, (f, e) in enumerate(zip(F_ens, F_err)):
        ax.text(k2, f + e + 0.004, f'{f:.3f}', ha='center', fontsize=7.5)
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.2, ls=':', lw=0.5)

    fig.suptitle('Ensemble fidelity — 500 trajectories, $T=300\\,$K, no renormalisation',
                 fontsize=10, fontweight='bold')
    if save:
        fig.savefig('fig4_ensemble.pdf'); fig.savefig('fig4_ensemble.png', dpi=200)
    return fig
