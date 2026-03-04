#!/usr/bin/env python3
"""
Plot PID vs MRAC adaptability comparison results.

Reads 4 CSV files produced by pid_vs_mrac_comparison and generates
a multi-panel figure showing velocity tracking, error, wrench, and
MRAC adaptive gain evolution.

Usage:
    python3 plot_comparison.py [csv_dir]
    (default csv_dir = current directory)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.8,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
})

# Color palette
C_PID = '#f85149'       # Red for PID
C_MRAC = '#58a6ff'      # Blue for MRAC
C_DESIRED = '#8b949e'   # Gray for desired/reference
C_REF = '#7ee787'       # Green for MRAC reference model
C_DAMAGE = '#d29922'    # Amber for damage line

DOF_NAMES = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']
DOF_UNITS = ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s']

# DOFs of interest (indices into the 6-DOF vectors)
ACTIVE_DOFS = [0, 2, 5]  # surge, heave, yaw
DAMAGE_TIME = 5.0


def load_csv(path):
    """Load a CSV file, return DataFrame."""
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        return None
    return pd.read_csv(path)


def add_damage_line(ax, damage_time):
    """Add a vertical dashed line at the damage onset time."""
    ax.axvline(x=damage_time, color=C_DAMAGE, linestyle='--', linewidth=1.0,
               alpha=0.7, label='Damage onset')


def main():
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

    # Load all data
    pid_normal = load_csv(os.path.join(csv_dir, 'pid_normal.csv'))
    mrac_normal = load_csv(os.path.join(csv_dir, 'mrac_normal.csv'))
    pid_damaged = load_csv(os.path.join(csv_dir, 'pid_damaged.csv'))
    mrac_damaged = load_csv(os.path.join(csv_dir, 'mrac_damaged.csv'))

    if any(d is None for d in [pid_normal, mrac_normal, pid_damaged, mrac_damaged]):
        print("ERROR: Not all CSV files found. Run pid_vs_mrac_comparison first.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1: Velocity Tracking Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(len(ACTIVE_DOFS), 2, figsize=(16, 10),
                                sharex=True, sharey='row')
    fig1.suptitle('Velocity Tracking: PID vs MRAC',
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    scenarios = [
        ('Normal Dynamics', pid_normal, mrac_normal, False),
        ('Damaged Dynamics (onset t=5s)', pid_damaged, mrac_damaged, True),
    ]

    for col, (title, pid_data, mrac_data, show_damage) in enumerate(scenarios):
        for row, dof_idx in enumerate(ACTIVE_DOFS):
            ax = axes1[row, col]
            dof_name = DOF_NAMES[dof_idx]
            dof_unit = DOF_UNITS[dof_idx]

            t = pid_data['time']
            des_col = f'des_{dof_name.lower()}'
            act_col = f'act_{dof_name.lower()}'

            # Desired
            ax.plot(t, pid_data[des_col], color=C_DESIRED, linewidth=1.0,
                    linestyle=':', alpha=0.8, label='Desired')

            # PID actual
            ax.plot(t, pid_data[act_col], color=C_PID, linewidth=1.2,
                    alpha=0.9, label='PID')

            # MRAC actual
            ax.plot(mrac_data['time'], mrac_data[act_col], color=C_MRAC,
                    linewidth=1.2, alpha=0.9, label='MRAC')

            # MRAC reference model
            ref_col = f'ref_{dof_name.lower()}'
            if ref_col in mrac_data.columns:
                ax.plot(mrac_data['time'], mrac_data[ref_col], color=C_REF,
                        linewidth=0.8, linestyle='--', alpha=0.7,
                        label='MRAC Ref Model')

            if show_damage:
                add_damage_line(ax, DAMAGE_TIME)

            ax.set_ylabel(f'{dof_name}\n({dof_unit})')
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(title, fontweight='bold')
            if row == len(ACTIVE_DOFS) - 1:
                ax.set_xlabel('Time (s)')
            if row == 0 and col == 0:
                ax.legend(loc='lower right', ncol=2)

    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(csv_dir, 'comparison_tracking.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2: Tracking Error Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(len(ACTIVE_DOFS), 2, figsize=(16, 10),
                                sharex=True)
    fig2.suptitle('Tracking Error: PID vs MRAC',
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    for col, (title, pid_data, mrac_data, show_damage) in enumerate(scenarios):
        for row, dof_idx in enumerate(ACTIVE_DOFS):
            ax = axes2[row, col]
            dof_name = DOF_NAMES[dof_idx]
            err_col = f'err_{dof_name.lower()}'

            t = pid_data['time']

            ax.plot(t, pid_data[err_col], color=C_PID, linewidth=1.0,
                    alpha=0.8, label='PID')
            ax.plot(mrac_data['time'], mrac_data[err_col], color=C_MRAC,
                    linewidth=1.0, alpha=0.8, label='MRAC')
            ax.axhline(y=0, color='#484f58', linewidth=0.5, linestyle='-')

            if show_damage:
                add_damage_line(ax, DAMAGE_TIME)

            ax.set_ylabel(f'{dof_name} Error')
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(title, fontweight='bold')
            if row == len(ACTIVE_DOFS) - 1:
                ax.set_xlabel('Time (s)')
            if row == 0 and col == 0:
                ax.legend(loc='upper right')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(csv_dir, 'comparison_error.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 3: Control Effort (Wrench) Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(len(ACTIVE_DOFS), 2, figsize=(16, 10),
                                sharex=True)
    fig3.suptitle('Control Effort (Wrench): PID vs MRAC',
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    wrench_names = ['wrench_surge', 'wrench_sway', 'wrench_heave',
                    'wrench_roll', 'wrench_pitch', 'wrench_yaw']
    wrench_units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

    for col, (title, pid_data, mrac_data, show_damage) in enumerate(scenarios):
        for row, dof_idx in enumerate(ACTIVE_DOFS):
            ax = axes3[row, col]
            w_col = wrench_names[dof_idx]
            dof_name = DOF_NAMES[dof_idx]

            t = pid_data['time']

            ax.plot(t, pid_data[w_col], color=C_PID, linewidth=0.8,
                    alpha=0.8, label='PID')
            ax.plot(mrac_data['time'], mrac_data[w_col], color=C_MRAC,
                    linewidth=0.8, alpha=0.8, label='MRAC')

            if show_damage:
                add_damage_line(ax, DAMAGE_TIME)

            ax.set_ylabel(f'{dof_name}\n({wrench_units[dof_idx]})')
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(title, fontweight='bold')
            if row == len(ACTIVE_DOFS) - 1:
                ax.set_xlabel('Time (s)')
            if row == 0 and col == 0:
                ax.legend(loc='upper right')

    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = os.path.join(csv_dir, 'comparison_wrench.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 4: MRAC Adaptive Gains Evolution
    # ═══════════════════════════════════════════════════════════════════════════
    fig4, axes4 = plt.subplots(len(ACTIVE_DOFS), 2, figsize=(16, 10),
                                sharex=True)
    fig4.suptitle('MRAC Adaptive Gains (Kx Diagonal): Normal vs Damaged',
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    mrac_scenarios = [
        ('Normal Dynamics', mrac_normal, False),
        ('Damaged Dynamics (onset t=5s)', mrac_damaged, True),
    ]

    kx_names = ['kx_surge', 'kx_sway', 'kx_heave',
                'kx_roll', 'kx_pitch', 'kx_yaw']

    for col, (title, mrac_data, show_damage) in enumerate(mrac_scenarios):
        for row, dof_idx in enumerate(ACTIVE_DOFS):
            ax = axes4[row, col]
            kx_col = kx_names[dof_idx]
            dof_name = DOF_NAMES[dof_idx]

            if kx_col in mrac_data.columns:
                ax.plot(mrac_data['time'], mrac_data[kx_col], color=C_MRAC,
                        linewidth=1.2, alpha=0.9)
                ax.fill_between(mrac_data['time'], 0, mrac_data[kx_col],
                                color=C_MRAC, alpha=0.15)

            if show_damage:
                add_damage_line(ax, DAMAGE_TIME)

            ax.axhline(y=0, color='#484f58', linewidth=0.5, linestyle='-')
            ax.set_ylabel(f'Kx_{dof_name}')
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(title, fontweight='bold')
            if row == len(ACTIVE_DOFS) - 1:
                ax.set_xlabel('Time (s)')

    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    out4 = os.path.join(csv_dir, 'comparison_adaptive_gains.png')
    fig4.savefig(out4, dpi=150, bbox_inches='tight')
    print(f"Saved: {out4}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 5: Summary — Error Norm Over Time (all 4 scenarios)
    # ═══════════════════════════════════════════════════════════════════════════
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig5.suptitle('Overall Tracking Error Norm: PID vs MRAC',
                  fontsize=16, fontweight='bold', color='white', y=0.98)

    all_err_cols = [f'err_{DOF_NAMES[i].lower()}' for i in range(6)]

    def error_norm(df):
        return np.sqrt(sum(df[c]**2 for c in all_err_cols))

    # Normal dynamics
    ax5a.plot(pid_normal['time'], error_norm(pid_normal), color=C_PID,
              linewidth=1.2, label='PID')
    ax5a.plot(mrac_normal['time'], error_norm(mrac_normal), color=C_MRAC,
              linewidth=1.2, label='MRAC')
    ax5a.set_title('Normal Dynamics', fontweight='bold')
    ax5a.set_ylabel('‖error‖')
    ax5a.legend()
    ax5a.grid(True, alpha=0.3)

    # Damaged dynamics
    ax5b.plot(pid_damaged['time'], error_norm(pid_damaged), color=C_PID,
              linewidth=1.2, label='PID')
    ax5b.plot(mrac_damaged['time'], error_norm(mrac_damaged), color=C_MRAC,
              linewidth=1.2, label='MRAC')
    add_damage_line(ax5b, DAMAGE_TIME)
    ax5b.set_title('Damaged Dynamics (onset t=5s)', fontweight='bold')
    ax5b.set_ylabel('‖error‖')
    ax5b.set_xlabel('Time (s)')
    ax5b.legend()
    ax5b.grid(True, alpha=0.3)

    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    out5 = os.path.join(csv_dir, 'comparison_summary.png')
    fig5.savefig(out5, dpi=150, bbox_inches='tight')
    print(f"Saved: {out5}")

    print(f"\n✓ All plots saved to {os.path.abspath(csv_dir)}/")


if __name__ == '__main__':
    main()
