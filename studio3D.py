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
#                      studio3D                     #
#             file con lo studio del 3D             #
#                                                   #
#####################################################

# -*- coding: utf-8 -*-

#--------------------------------
#    Aggiungo moduli aggiuntivi
#--------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -
# Moduli propri
# -

from galton import Galton3D
from util import NumPalAdeg

###############
#  STUDIO 3D  #
###############

def studio3D():
    print("Hai avviato lo studio della macchina di Galton 3D\n")
    print("Maggiori informazioni sono fornite nel file README.\n\n")
    
    print("-----\n")
    
    
    print("PARTE 1")
    
    studio3Dpt1()
    
    print("\n----\n")
    
    
    print("PARTE 2")
    
    studio3Dpt2()
    
    print("\n----\n")
    
    
    print("PARTE 3")
    
    studio3Dpt3()
    
    print("\n-----")
        



#-----------------------
#   Studio 3D parte 1
#-----------------------

def studio3Dpt1():
    """
    Si varia il numero di passi, mantenendo inviati probX e probY.
    
    """
    print(f"\nSi varia il numero di passi.\n")
    passi = np.array([10, 30, 50, 80, 100, 200, 500, 800])
    
    px = 0.5
    py = 0.5
    
    
    print(f"Probabilità di andare a DX lungo l'asse X: {px}")
    print(f"Probabilità di andare a DX lungo l'asse Y: {py}")
    
    print("Passi scelti per eseguire la simulazione:")
    [print("*", p) for p in passi]
    
    chi2BinX = np.empty((len(passi), 2))
    chi2BinY = np.empty((len(passi), 2))
    
    chi2GauX = np.empty((len(passi), 2))
    chi2GauY = np.empty((len(passi), 2))
    
    parBinX = np.empty((len(passi), 3))
    parBinY = np.empty((len(passi), 3))
    
    parGauX = np.empty((len(passi), 3))
    parGauY = np.empty((len(passi), 3))
    
    gdlBinX = np.empty((len(passi)))
    gdlBinY = np.empty((len(passi)))
    
    gdlGauX = np.empty((len(passi)))
    gdlGauY = np.empty((len(passi)))

    for i in range(len(passi)):
        print(f"Numero di passi: {passi[i]}")
        
        G = Galton3D(passi[i], NumPalAdeg(passi[i], px, 0.01, py), px, py, False)
        
        chi2BinX[i] = G.chi2BinX, G.chi2RidBinX
        chi2BinY[i] = G.chi2BinY, G.chi2RidBinY
        chi2GauX[i] = G.chi2GauX, G.chi2RidGauX
        chi2GauY[i] = G.chi2GauY, G.chi2RidGauY
        
        parBinX[i] = G.parBinX
        parBinY[i] = G.parBinY
        parGauX[i] = G.parGauX
        parGauY[i] = G.parGauY
        
        gdlBinX[i] = G.gdlBinX
        gdlBinY[i] = G.gdlBinY
        gdlGauX[i] = G.gdlGauX
        gdlGauY[i] = G.gdlGauY
    
    print("\nTabella 1: Parametri Binomiali X\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("Passi", "n", "p", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(passi)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            passi[i], parBinX[i][0], parBinX[i][1], gdlBinX[i], chi2BinX[i][1]
        ))
    
    print("\nTabella 2: Parametri Binomiali Y\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("Passi", "n", "p", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(passi)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            passi[i], parBinY[i][0], parBinY[i][1], gdlBinY[i], chi2BinY[i][1]
        ))
    
    print("\nTabella 3: Parametri Gaussiani X\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("Passi", "µ", "𝜎", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(passi)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            passi[i], parGauX[i][0], parGauX[i][1], gdlGauX[i], chi2GauX[i][1]
        ))
    
    print("\nTabella 4: Parametri Gaussiani Y\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("Passi", "µ", "𝜎", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(passi)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            passi[i], parGauY[i][0], parGauY[i][1], gdlGauY[i], chi2GauY[i][1]
        ))
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].plot(passi, chi2BinX[:, 1], 'o-')
    axs[0, 0].set_title('Binomiale X')
    axs[0, 0].set_xlabel('Numero di passi')
    axs[0, 0].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[0, 1].plot(passi, chi2BinY[:, 1], 'o-')
    axs[0, 1].set_title('Binomiale Y')
    axs[0, 1].set_xlabel('Numero di passi')
    axs[0, 1].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[1, 0].plot(passi, chi2GauX[:, 1], 'o-')
    axs[1, 0].set_title('Gaussiana X')
    axs[1, 0].set_xlabel('Numero di passi')
    axs[1, 0].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[1, 1].plot(passi, chi2GauY[:, 1], 'o-')
    axs[1, 1].set_title('Gaussiana Y')
    axs[1, 1].set_xlabel('Numero di passi')
    axs[1, 1].set_ylabel('$\\tilde{\\chi}^2$')
    
    plt.suptitle(f"Variazione del $\\tilde{{\\chi}}^2$ in funzione del numero di passi\nprobX={px}, probY={py}")
    plt.tight_layout()
    plt.show()
        
#-----------------------
#   Studio 3D parte 2
#-----------------------

def studio3Dpt2():
    """
    Si varia probX, mantenendo inviati il numero di passi e probY.
    
    """
    
    print(f"\nSi varia la probabilità di andare a DX lungo l'asse X.\n")
    passi = 400
    
    px = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    py = 0.5
    
    print(f"Numero di passi scelti: {passi}")
    print(f"Probabilità di andare a DX lungo l'asse Y: {py}")
    
    print("Probabiità di andare a DX lungo l'asse X scelte per eseguire la simulazione:")
    [print("*", x) for x in px]
    
    chi2BinX = np.empty((len(px), 2))
    chi2BinY = np.empty((len(px), 2))
    
    chi2GauX = np.empty((len(px), 2))
    chi2GauY = np.empty((len(px), 2))
    
    parBinX = np.empty((len(px), 3))
    parBinY = np.empty((len(px), 3))
    
    parGauX = np.empty((len(px), 3))
    parGauY = np.empty((len(px), 3))
    
    gdlBinX = np.empty((len(px)))
    gdlBinY = np.empty((len(px)))
    
    gdlGauX = np.empty((len(px)))
    gdlGauY = np.empty((len(px)))

    for i in range(len(px)):
        print(f"Probabilità probX: {px[i]}")
        
        G = Galton3D(passi, NumPalAdeg(passi, px[i], 0.01, py), px[i], py, False)
        
        chi2BinX[i] = G.chi2BinX, G.chi2RidBinX
        chi2BinY[i] = G.chi2BinY, G.chi2RidBinY
        chi2GauX[i] = G.chi2GauX, G.chi2RidGauX
        chi2GauY[i] = G.chi2GauY, G.chi2RidGauY
        
        parBinX[i] = G.parBinX
        parBinY[i] = G.parBinY
        parGauX[i] = G.parGauX
        parGauY[i] = G.parGauY
        
        gdlBinX[i] = G.gdlBinX
        gdlBinY[i] = G.gdlBinY
        gdlGauX[i] = G.gdlGauX
        gdlGauY[i] = G.gdlGauY
        
    print("\nTabella 1: Parametri Binomiali X\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probX", "n", "p", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(px)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            px[i], parBinX[i][0], parBinX[i][1], gdlBinX[i], chi2BinX[i][1]
        ))
    
    print("\nTabella 2: Parametri Binomiali Y\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probX", "n", "p", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(px)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            px[i], parBinY[i][0], parBinY[i][1], gdlBinY[i], chi2BinY[i][1]
        ))
    
    print("\nTabella 3: Parametri Gaussiani X\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probX", "µ", "𝜎", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(px)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            px[i], parGauX[i][0], parGauX[i][1], gdlGauX[i], chi2GauX[i][1]
        ))
    
    print("\nTabella 4: Parametri Gaussiani Y\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probX", "µ", "𝜎", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(px)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            px[i], parGauY[i][0], parGauY[i][1], gdlGauY[i], chi2GauY[i][1]
        ))
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].plot(px, chi2BinX[:, 1], 'o-')
    axs[0, 0].set_title('Binomiale X')
    axs[0, 0].set_xlabel('Probabilità di andare a DX lungo X')
    axs[0, 0].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[0, 1].plot(px, chi2BinY[:, 1], 'o-')
    axs[0, 1].set_title('Binomiale Y')
    axs[0, 1].set_xlabel('Probabilità di andare a DX lungo X')
    axs[0, 1].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[1, 0].plot(px, chi2GauX[:, 1], 'o-')
    axs[1, 0].set_title('Gaussiana X')
    axs[1, 0].set_xlabel('Probabilità di andare a DX lungo X')
    axs[1, 0].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[1, 1].plot(px, chi2GauY[:, 1], 'o-')
    axs[1, 1].set_title('Gaussiana Y')
    axs[1, 1].set_xlabel('Probabilità di andare a DX lungo X')
    axs[1, 1].set_ylabel('$\\tilde{\\chi}^2$')
    
    plt.suptitle(f"Variazione del $\\tilde{{\\chi}}^2$ in funzione della probabilità di spostamento lungo X\nnPassi={passi}, probY={py}")
    plt.tight_layout()
    plt.show()
        
    

#-----------------------
#   Studio 3D parte 3
#-----------------------

def studio3Dpt3():
    """
    Si varia probY, mantenendo invariati probY e il numero di passi.
    
    """
    
    
    print(f"\nSi varia la probabilità di andare a DX lungo l'asse Y.\n")
    passi = 400
    
    px = 0.5
    py = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    
    print(f"Numero di passi scelti: {passi}")
    print(f"Probabilità di andare a DX lungo l'asse Y: {px}")
    
    print("Probabiità di andare a DX lungo l'asse X scelte per eseguire la simulazione:")
    [print("*", x) for x in py]
    
    chi2BinX = np.empty((len(py), 2))
    chi2BinY = np.empty((len(py), 2))
    
    chi2GauX = np.empty((len(py), 2))
    chi2GauY = np.empty((len(py), 2))
    
    parBinX = np.empty((len(py), 3))
    parBinY = np.empty((len(py), 3))
    
    parGauX = np.empty((len(py), 3))
    parGauY = np.empty((len(py), 3))
    
    gdlBinX = np.empty((len(py)))
    gdlBinY = np.empty((len(py)))
    
    gdlGauX = np.empty((len(py)))
    gdlGauY = np.empty((len(py)))
    
    for i in range(len(py)):
        print(f"Probabilità probY: {py[i]}")
        
        G = Galton3D(passi, NumPalAdeg(passi, px, 0.01, py[i]), px, py[i], False)
        
        chi2BinX[i] = G.chi2BinX, G.chi2RidBinX
        chi2BinY[i] = G.chi2BinY, G.chi2RidBinY
        chi2GauX[i] = G.chi2GauX, G.chi2RidGauX
        chi2GauY[i] = G.chi2GauY, G.chi2RidGauY
        
        parBinX[i] = G.parBinX
        parBinY[i] = G.parBinY
        parGauX[i] = G.parGauX
        parGauY[i] = G.parGauY
        
        gdlBinX[i] = G.gdlBinX
        gdlBinY[i] = G.gdlBinY
        gdlGauX[i] = G.gdlGauX
        gdlGauY[i] = G.gdlGauY
    
    print("\nTabella 1: Parametri Binomiali X\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probY", "n", "p", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(py)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            py[i], parBinX[i][0], parBinX[i][1], gdlBinX[i], chi2BinX[i][1]
        ))
    
    print("\nTabella 2: Parametri Binomiali Y\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probY", "n", "p", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(py)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            py[i], parBinY[i][0], parBinY[i][1], gdlBinY[i], chi2BinY[i][1]
        ))
    
    print("\nTabella 3: Parametri Gaussiani X\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probY", "µ", "𝜎", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(py)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            py[i], parGauX[i][0], parGauX[i][1], gdlGauX[i], chi2GauX[i][1]
        ))
    
    print("\nTabella 4: Parametri Gaussiani Y\n")
    print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("probY", "µ", "𝜎", "GdL", "χ²rid"))
    print("-" * 60)
    for i in range(len(py)):
        print("{:<8} {:<10.5f} {:<10.5f} {:<10.5f} {:<10.5f}".format(
            py[i], parGauY[i][0], parGauY[i][1], gdlGauY[i], chi2GauY[i][1]
        ))


    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].plot(py, chi2BinX[:, 1], 'o-')
    axs[0, 0].set_title('Binomiale X')
    axs[0, 0].set_xlabel('Probabilità di andare a DX lungo Y')
    axs[0, 0].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[0, 1].plot(py, chi2BinY[:, 1], 'o-')
    axs[0, 1].set_title('Binomiale Y')
    axs[0, 1].set_xlabel('Probabilità di andare a DX lungo Y')
    axs[0, 1].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[1, 0].plot(py, chi2GauX[:, 1], 'o-')
    axs[1, 0].set_title('Gaussiana X')
    axs[1, 0].set_xlabel('Probabilità di andare a DX lungo Y')
    axs[1, 0].set_ylabel('$\\tilde{\\chi}^2$')
    
    axs[1, 1].plot(py, chi2GauY[:, 1], 'o-')
    axs[1, 1].set_title('Gaussiana Y')
    axs[1, 1].set_xlabel('Probabilità di andare a DX lungo Y')
    axs[1, 1].set_ylabel('$\\tilde{\\chi}^2$')
    
    plt.suptitle(f"Variazione del $\\tilde{{\\chi}}^2$ in funzione della probabilità di spostamento lungo Y\nnPassi={passi}, probX={px}")
    plt.tight_layout()
    plt.show()