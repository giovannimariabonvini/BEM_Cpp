#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

def parse_convergence_table(filename):
    """
    Parses a table with columns (example):
      cycle  cells dofs Mesh_size   L2(phi) factor Linfty(phi) factor L2(phi_n) factor Linfty(phi_n) factor Linfty(alpha) factor
    We'll extract: cycle, Mesh_size, L2(phi), Linfty(phi), L2(phi_n), Linfty(phi_n).
    """
    cycles   = []
    meshsize = []
    l2phi    = []
    linfphi  = []
    l2phin   = []
    linfphin = []

    with open(filename, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        # Skip headers, empty lines, lines with "Mesh_size" or "---"
        if not line.strip():
            continue
        if ("cycle" in line) or ("Mesh_size" in line) or ("---" in line):
            continue

        tokens = line.split()
        # Example tokens:
        #   0->cycle, 1->cells, 2->dofs, 3->Mesh_size,
        #   4->L2(phi), 5->factor, 6->Linfty(phi),7->factor,
        #   8->L2(phi_n),9->factor,10->Linfty(phi_n),11->factor,
        #   12->Linfty(alpha),13->factor
        cyc  = int(tokens[0])
        ms   = float(tokens[3])
        l2p  = float(tokens[4])
        linp = float(tokens[6])
        l2pn = float(tokens[8])
        linpn= float(tokens[10])

        cycles.append(cyc)
        meshsize.append(ms)
        l2phi.append(l2p)
        linfphi.append(linp)
        l2phin.append(l2pn)
        linfphin.append(linpn)

    return cycles, meshsize, l2phi, linfphi, l2phin, linfphin

def main():
    datafile = "convergence_table.txt"

    # Parse the file
    (cycles, meshsize, l2phi, linfphi, l2phin, linfphin) = parse_convergence_table(datafile)

    # Reference: first row => largest mesh
    l2phi_ref    = l2phi[0]
    linfphi_ref  = linfphi[0]
    l2phin_ref   = l2phin[0]
    linfphin_ref = linfphin[0]

    # Normalize the data

    if(l2phi_ref != 0):
        norm_l2phi = [val / l2phi_ref for val in l2phi]
    else:
        norm_l2phi = [val for val in l2phi]

    if(linfphi_ref != 0):
        norm_linfphi = [val / linfphi_ref for val in linfphi]
    else:
        norm_linfphi = [val for val in linfphi]

    if(l2phin_ref != 0):
        norm_l2phin = [val / l2phin_ref for val in l2phin]
    else:
        norm_l2phin = [val for val in l2phin]

    if(linfphin_ref != 0):
        norm_linfphin = [val / linfphin_ref for val in linfphin]
    else:
        norm_linfphin = [val for val in linfphin]

    # If error is too small means that it is the boundary condition assigned, so set to zero
    if(l2phi_ref < 1e-12):
        norm_l2phi = [0 for val in l2phi]
    if(linfphi_ref < 1e-12):
        norm_linfphi = [0 for val in linfphi]
    if(l2phin_ref < 1e-12):
        norm_l2phin = [0 for val in l2phin]
    if(linfphin_ref < 1e-12):
        norm_linfphin = [0 for val in linfphin]


    #
    # 1) Plot L2 errors (phi and phi_n) with reference lines for orders 1,2,3
    #
    plt.figure(figsize=(7,5))

    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    # Plot our data
    plt.plot(meshsize, norm_l2phi,   "o-", label="L2(phi) normalized")       # COMMENT THIS LINE TO NOT PLOT L2(phi)
    plt.plot(meshsize, norm_l2phin,  "s-", label="L2(phi_n) normalized")

    # X axis in log scale with decimal formatting
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    # Y axis in log scale
    plt.yscale("log")

    # Add reference lines for orders p=1,2,3 => y = x^p
    # We'll define a smooth range of x from min->max
    x_vals = np.logspace(np.log10(min(meshsize)), np.log10(max(meshsize)), 50)
    orders = [0.5, 1, 2, 3]
    linestyles = [":","dotted", "dashdot", "dashed"]  # each string is a valid Matplotlib style
    for i, p in enumerate(orders):
        y_vals = x_vals ** (p)  # slope ~ -p
        plt.plot(x_vals, y_vals/y_vals[-1],
                color='black',
                linestyle=linestyles[i],
                label=f"Order {p}")

    plt.xlabel("Normalized mesh size (h/h0)")
    plt.ylabel("Normalized L2 error")
    plt.title("Convergence: L2 errors vs. mesh size (normalized)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig("convergence_L2.png", dpi=150)

    #
    # 2) Plot Linfty errors (phi and phi_n) with the same reference lines
    #
    plt.figure(figsize=(7,5))

    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    # Plot our data
    plt.plot(meshsize, norm_linfphi,   "o-", label="Linfty(phi) normalized")
    plt.plot(meshsize, norm_linfphin,  "s-", label="Linfty(phi_n) normalized")

    # X axis in log scale with decimal formatting
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    # Y axis in log scale
    plt.yscale("log")

    # We reuse the same x_vals for the lines
    orders = [0.5, 1, 2, 3]
    linestyles = [":","dotted", "dashdot", "dashed"]  # each string is a valid Matplotlib style
    for i, p in enumerate(orders):
        y_vals = x_vals ** (p)  # slope ~ -p
        plt.plot(x_vals, y_vals/y_vals[-1],
                color='black',
                linestyle=linestyles[i],
                label=f"Order {p}")

    plt.xlabel("Normalized mesh size (h/h0)")
    plt.ylabel("Normalized Linfty error")
    plt.title("Convergence: Linfty errors vs. mesh size (normalized)")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig("convergence_Linf.png", dpi=150)

    # plt.show()

if __name__ == "__main__":
    main()
