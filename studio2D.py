#####################################################
#                                                   #
#         Università degli Studi di Perugia         #
#            Laurea Triennale in Fisica             #
#                                                   #
#        Metodi Computazionali per la Fisica        #
#             Anno accademico 2024/2025             #
#                                                   #
#---------------------------------------------------#
#                                                   #
#        Elaborato finale di Filippo Tintori        #
#                                                   #
#              GitHub: filippo-tintori              #
#   https://github.com/filippo-tintori/ProgettoMCF  #
#                                                   #
#---------------------------------------------------#
#                                                   #
#                      studio2D                     #
#             file con lo studio del 2D             #
#                                                   #
#####################################################

# -*- coding: utf-8 -*-

#--------------------------------
#    Aggiungo moduli aggiuntivi
#--------------------------------

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import chi2

# -
# Moduli propri
# -

from galton import Galton2D
from util import NumPalAdeg
from util import gaussiana
from util import binomiale

#------------------------------------------
#    Modifica dei plot in stile LaTeX
#------------------------------------------

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

###############
#  STUDIO 2D  #
###############

def studio2D():
    print("Hai avviato lo studio della macchina di Galton 2D.\n")
    print("Verranno eseguite le simulazioni di diverse configurazioni di macchine di Galton.")
    print("Maggiori informazioni sono fornite nel file README.\n\n")
    
    
    print("-----\n")
    
    print("PARTE 1")
    
    studio2Dpt1()


    print("\n----\n")
    
    print("PARTE 2")
    
    studio2Dpt2()
    
    
    print("\n----\n")
     
    print("PARTE 3")
    
    studio2Dpt3()
    
    print("\n-----")


#-----------------------
#   Studio 2D parte 1
#-----------------------

def studio2Dpt1():
    """
    Si ripete la stessa simulazione per 1000 volte.
    
    """
    ripet = 1000
    nbin = 7
    prob = 0.5
    passi = 20
    
    n = NumPalAdeg(passi, prob, 0.01)
    
    studio1 = np.empty(ripet)
    Bin = np.empty(ripet)
    
    for i in tqdm(range(ripet)):
        studio1[i] = Galton2D(passi, n, prob, False).parGau[0]
    
    hist, binBordi = np.histogram(studio1, bins=nbin)
    binCentri = (binBordi[:-1] + binBordi[1:]) / 2
    
    par, _ = curve_fit(gaussiana, binCentri, hist, p0=[np.mean(studio1), np.std(studio1),np.max(hist)])
    
    mu, sigma, amp = par
    
    xFit = np.linspace(binCentri[0], binCentri[-1], 100)
    yFit = gaussiana(xFit, mu, sigma, amp)
    
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.hist(studio1, bins=nbin, color='blue', alpha=0.7, edgecolor='black', label='Media $\\mu$')
    
    plt.title("Senza fit")
    plt.xlabel("$\\mu$")
    plt.ylabel("Frequenza")
    plt.legend()
    
    
    plt.subplot(1, 2, 2)
    
    plt.hist(studio1, bins=nbin, color='blue', alpha=0.7, edgecolor='black', label='Media $\\mu$')
    plt.plot(xFit, yFit, 'r-', label=f"Fit Gaussiano\n$\\mu={mu:.2f}, \\sigma={sigma:.4f}$")
    plt.axvline(prob*20, color='coral', linestyle='--', label='Valore teorico $\\mu$')
    
    plt.title("Con fit")
    plt.xlabel("$\\mu$")
    plt.ylabel("Frequenza")
    plt.legend()
    
    # ---
    # print
    # ---
    
    t = abs(mu - prob*passi) / sigma
    
    P = 2 * (1 - norm.cdf(t)) # 2 per la simmetria
    
    print("\nRISULTATI\n")
    
    print("Valore teorico:")
    print(f"  p = {prob:.2f}")
    print(f"  µ = {prob*passi:.2f}")
    
    print("\nFit gaussiana:")
    print(f"  µ = {mu:.4f}")
    print(f"  𝜎 = {sigma:.4f}")
    
    print("Probabilità gaussiana:")
    print(f"  t = {t:.4f}")
    print(f"  P(entro t𝜎) = {(1-P):.3f} = {(1-P)*100:.3f} %")
    print(f"  P(fuori da t𝜎) = {(P):.3f} = {(P)*100:.3f} %")
    
    # ---
    # plot
    # ---
    
    plt.suptitle(f'Distribuzione delle medie stimate $\\mu$\nnPassi={passi},  nPalle={n},  probDX={prob}')
    plt.tight_layout()
    plt.show()




#-----------------------
#   Studio 2D parte 2
#-----------------------


def studio2Dpt2():
    """
    Si varia il numero di passi, mantenendo la probabilità invariata.
    
    """
    passi = np.array([5, 10, 20,  50, 100, 200, 300, 400, 500, 800, 1000])
    
    print("Probabilità di andare a DX scelta: 0.5")
    
    print("Passi scelti per eseguire la simulazione:")
    [print("*", p) for p in passi]
    
    chi2Bin = np.empty((len(passi), 2))
    chi2Gau = np.empty((len(passi), 2))
    
    parBin = np.empty((len(passi), 3))
    parGau = np.empty((len(passi), 3))
    
    gdlBin = np.empty((len(passi)))
    gdlGau = np.empty((len(passi)))
    
    
    for i in tqdm(range(len(passi))):
        G = Galton2D(passi[i], NumPalAdeg(passi[i], 0.5, 0.01), 0.5, False)
        parBin[i] = G.parBin
        parGau[i] = G.parGau
        gdlBin[i] = G.gdlBin
        gdlGau[i] = G.gdlGau
        chi2Bin[i] = G.chi2Bin, G.chi2RidBin
        chi2Gau[i] = G.chi2Gau, G.chi2RidGau
    
    print("\n" + "-" * 45)
    print("Test del χ² per la Binomiale\n")
    print(f"{'Indice':<6} {'χ²':>7} {'|':^5} {'GdL':>6} {'|':^5} {'χ²rid':>7}{'|'}")
    print("-" * 42)
    for i in range(len(gdlBin)):
        print(f"{i:<6} {chi2Bin[i][0]:>7.2f} {'|':^5} {gdlBin[i]:>6.0f} {'|':^5} {chi2Bin[i][1]:>7.3f}{'|'}")

    print("\n" + "-" * 45)
    print("\nTest del χ² per la Gaussiana\n")
    print(f"{'Indice':<6} {'χ²':>7} {'|':^5} {'GdL':>6} {'|':^5} {'χ²rid':>7}{'|'}")
    print("-" * 42)
    for i in range(len(gdlGau)):
        print(f"{i:<6} {chi2Gau[i][0]:>7.2f} {'|':^5} {gdlGau[i]:>6.0f} {'|':^5} {chi2Gau[i][1]:>7.3f}{'|'}")

    print("\n" + "-" * 45 + "\n")
    
    
    # -----
    # GRAFICO DI CONFRONTO TRA PARAMETRI
    # -----
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    
    axes[0, 0].plot(passi, parGau[:, 0], marker='o', label='$\\mu$', color='skyblue')
    axes[0, 0].set_title('Gaussiana: Evoluzione di $\\mu$')
    axes[0, 0].set_xlabel('Numero di passi')
    axes[0, 0].set_ylabel('$\\mu$')
    axes[0, 0].legend()
    
    axes[0, 1].plot(passi, parGau[:, 1], marker='s', label='$\\sigma$', color='r')
    axes[0, 1].set_title('Gaussiana: Evoluzione di $\\sigma$')
    axes[0, 1].set_xlabel('Numero di passi')
    axes[0, 1].set_ylabel('$\\sigma$')
    axes[0, 1].legend()
    
    axes[0, 2].plot(passi, parGau[:, 2], marker='d', label='amp', color='g')
    axes[0, 2].set_title('Gaussiana: Evoluzione di amp')
    axes[0, 2].set_xlabel('Numero di passi')
    axes[0, 2].set_ylabel('amp')
    axes[0, 2].legend()
    
    # Grafici della distribuzione binomiale
    axes[1, 0].plot(passi, parBin[:, 0], marker='o', label='$n$', color='blue')
    axes[1, 0].set_title('Binomiale: Evoluzione di $n$')
    axes[1, 0].set_xlabel('Numero di passi')
    axes[1, 0].set_ylabel('$n$')
    axes[1, 0].legend()
    
    axes[1, 1].plot(passi, parBin[:, 1], marker='s', label='$p$', color='orange')
    axes[1, 1].set_title('Binomiale: Evoluzione di $p$')
    axes[1, 1].set_xlabel('Numero di passi')
    axes[1, 1].set_ylabel('$p$')
    axes[1, 1].legend()
    
    axes[1, 2].plot(passi, parBin[:, 2], marker='d', label='amp', color='lightgreen')
    axes[1, 2].set_title('Binomiale: Evoluzione di amp')
    axes[1, 2].set_xlabel('Numero di passi')
    axes[1, 2].set_ylabel('amp')
    axes[1, 2].legend()
    
    plt.suptitle(f"Evoluzione dei parametri al variare del numero di passi\nprobDX={0.5}")
    plt.tight_layout()
    plt.show()
    
    # -----
    # GRAFICO DI CONFRONTO TRA DISTRIBUZIONI
    # -----
    
    xVal = np.linspace(0, passi[-1], 10000)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    
    for i in range(len(passi)):
        #binomiale
        ax[0].plot(xVal, binomiale(xVal, parBin[i][0], parBin[i][1], parBin[i][2]), label=f"Binomiale $n$={passi[i]}")
        ax[0].set_title("Distribuzione Binomiale")
        ax[0].set_xlabel("$n$")
        ax[0].set_ylabel("Frequenza")
        ax[0].legend()

        #gaussiana
        ax[1].plot(xVal, gaussiana(xVal, parGau[i][0], parGau[i][1], parGau[i][2]), label=f"Gaussiana $n$={passi[i]}")
        ax[1].set_title("Distribuzione Gaussiana")
        ax[1].set_xlabel("$n$")
        ax[1].set_ylabel("Frequenza")
        ax[1].legend() 
    
    plt.suptitle("Evoluzione delle distribuzioni al variare del numero di passi\nprobDX=0.5")
    plt.tight_layout()
    plt.show()


#-----------------------
#   Studio 2D parte 3
#-----------------------


def studio2Dpt3():
    """
    Si varia il la probabilità, mantenendo invariati il numero di passi.
    
    """
    
    passi = 500
    prob = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    
    print(f"Numero di passi scelti: {passi}")
    
    print("Probabilità di andare a DX scelte per eseguire la simulazione:")
    [print("*", p) for p in prob]
    
    chi2Bin = np.empty((len(prob), 2))
    chi2Gau = np.empty((len(prob), 2))
    
    parBin = np.empty((len(prob), 3))
    parGau = np.empty((len(prob), 3))
    
    gdlBin = np.empty((len(prob)))
    gdlGau = np.empty((len(prob)))
    
    for i in tqdm(range(len(prob))):
        G = Galton2D(passi, NumPalAdeg(passi, prob[i], 0.01), prob[i], False)
        parBin[i] = G.parBin
        parGau[i] = G.parGau
        gdlBin[i] = G.gdlBin
        gdlGau[i] = G.gdlGau
        chi2Bin[i] = G.chi2Bin, G.chi2RidBin
        chi2Gau[i] = G.chi2Gau, G.chi2RidGau
    
    print("\n" + "-" * 45)
    print("Test del χ² per la Binomiale\n")
    print(f"{'Indice':<6} {'χ²':>7} {'|':^5} {'GdL':>6} {'|':^5} {'χ²rid':>7}{'|'}")
    print("-" * 42)
    for i in range(len(gdlBin)):
        print(f"{i:<6} {chi2Bin[i][0]:>7.2f} {'|':^5} {gdlBin[i]:>6.0f} {'|':^5} {chi2Bin[i][1]:>7.3f}{'|'}")

    print("\n" + "-" * 45)
    print("\nTest del χ² per la Gaussiana\n")
    print(f"{'Indice':<6} {'χ²':>7} {'|':^5} {'GdL':>6} {'|':^5} {'χ²rid':>7}{'|'}")
    print("-" * 42)
    for i in range(len(gdlGau)):
        print(f"{i:<6} {chi2Gau[i][0]:>7.2f} {'|':^5} {gdlGau[i]:>6.0f} {'|':^5} {chi2Gau[i][1]:>7.3f}{'|'}")

    print("\n" + "-" * 45 + "\n")
    
    
    # -----
    # GRAFICO DI CONFRONTO TRA PARAMETRI
    # -----
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    
    # distribuzione gaussiana
    axes[0, 0].plot(prob, parGau[:, 0], marker='o', label=f'$\\mu$', color='skyblue')
    axes[0, 0].set_title(f'Gaussiana: Evoluzione di $\\mu$')
    axes[0, 0].set_xlabel('Probabilità di andare a DX')
    axes[0, 0].set_ylabel(f'$\\mu$')
    axes[0, 0].legend()
    
    axes[0, 1].plot(prob, parGau[:, 1], marker='s', label=f'$\\sigma$', color='r')
    axes[0, 1].set_title(f'Gaussiana: Evoluzione di $\\sigma$')
    axes[0, 1].set_xlabel('Probabilità di andare a DX')
    axes[0, 1].set_ylabel(f'$\\sigma$')
    axes[0, 1].legend()
    
    axes[0, 2].plot(prob, parGau[:, 2], marker='d', label='amp', color='g')
    axes[0, 2].set_title('Gaussiana: Evoluzione di amp')
    axes[0, 2].set_xlabel('Probabilità di andare a DX')
    axes[0, 2].set_ylabel('amp')
    axes[0, 2].legend()
    
    # distribuzione binomiale
    axes[1, 0].plot(prob, parBin[:, 0], marker='o', label=f'$n$', color='blue')
    axes[1, 0].set_title(f'Binomiale: Evoluzione di $n$')
    axes[1, 0].set_xlabel('Probabilità di andare a DX')
    axes[1, 0].set_ylabel(f'$n$')
    axes[1, 0].legend()
    
    axes[1, 1].plot(prob, parBin[:, 1], marker='s', label=f'$p$', color='orange')
    axes[1, 1].set_title(f'Binomiale: Evoluzione di $p$')
    axes[1, 1].set_xlabel('Probabilità di andare a DX')
    axes[1, 1].set_ylabel(f'$p$')
    axes[1, 1].legend()
    
    axes[1, 2].plot(prob, parBin[:, 2], marker='d', label='amp', color='lightgreen')
    axes[1, 2].set_title('Binomiale: Evoluzione di amp')
    axes[1, 2].set_xlabel('Probabilità di andare a DX')
    axes[1, 2].set_ylabel('amp')
    axes[1, 2].legend()
    
    plt.suptitle(f"Evoluzione dei parametri al variare della probabilità di andare a DX\nnPassi={passi}")
    plt.tight_layout()
    plt.show()
    
    # -----
    # GRAFICO DI CONFRONTO TRA DISTRIBUZIONI
    # -----
    
    xVal = np.linspace(0, passi, 10000)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    
    for i in range(len(prob)):
        #binomiale
        ax[0].plot(xVal, binomiale(xVal, parBin[i][0], parBin[i][1], parBin[i][2]), label=f"Binomiale $p$={prob[i]}")
        ax[0].set_title("Distribuzione Binomiale")
        ax[0].set_xlabel(f"$n$")
        ax[0].set_ylabel("Frequenza")
        ax[0].legend()

        #gaussiana
        ax[1].plot(xVal, gaussiana(xVal, parGau[i][0], parGau[i][1], parGau[i][2]), label=f"Gaussiana $p$={prob[i]}")
        ax[1].set_title("Distribuzione Gaussiana")
        ax[1].set_xlabel(f"$n$")
        ax[1].set_ylabel("Frequenza")
        ax[1].legend() 
    
    plt.suptitle(f"Evoluzione delle distribuzioni al variare della probabilità di andare a DX\nnPassi={passi}")
    plt.tight_layout()
    plt.show()