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
#                    studio3Dcorr                   #
#        file con lo studio del 3D correlato        #
#                                                   #
#####################################################

# -*- coding: utf-8 -*-

#--------------------------------
#    Aggiungo moduli aggiuntivi
#--------------------------------

import numpy as np

# -
# Moduli propri
# -

from galton import Galton3Dcorr
from util import NumPalAdeg


####################
#  STUDIO 3D CORR  #
####################

def studio3Dcorr():
    
    # 1. Nessuna correlazione (indipendenza)
    M1 = np.array([[1, 0.1],
                   [0.1, 1]])

    # 2. Correlazione positiva forte
    M2 = np.array([[1, 0.9],
                   [0.9, 1]])

    # 3. Correlazione negativa forte
    M3 = np.array([[1, -0.7],
                   [-0.7, 1]])

    # 4. Correlazione moderata
    M4 = np.array([[1, 0.5],
                   [0.5, 1]])
    
    
    M = np.array([M1, M2, M3, M4])
    
    print("---\n")
    print("Hai avviato lo studio della macchina di Galton 3D correlata\n")
    print("Maggiori informazioni sono fornite nel file README.\n\n")
    
    px = 0.8
    py = 0.3
    nPassi = 100
    nPalle = NumPalAdeg(nPassi, px, 0.01, py)
    
    print(f"Numero di palle adeguato: {nPalle}")
    
    parBinX = np.empty((len(M), 3))  # (n, p, amp) per X
    parGauX = np.empty((len(M), 3))  # (mu, sigma, amp) per X
    
    parBinY = np.empty((len(M), 3))  # (n, p, amp) per Y
    parGauY = np.empty((len(M), 3))  # (mu, sigma, amp) per Y
    
    matr = np.empty((len(M), 2, 2))  # Matrici di correlazione inserite
    matrStim = np.empty((len(M), 2, 2))  # Matrici di correlazione stimate
    
    
    for i in range(len(M)):
        G = Galton3Dcorr(nPassi, nPalle, px, py, M[i], False)
        matr[i] = G.matriceCorrelazione
        matrStim[i] = G.matriceCorrelazioneStimata
        parBinX[i] = G.parBinX
        parBinY[i] = G.parBinY
        parGauX[i] = G.parGauX
        parGauY[i] = G.parGauY

    print(f" Numero di passi = {nPassi}")
    print(f" ProbX = {px}")
    print(f" ProbY = {py}")
    print("╔═══╦════════╦════════╦═════════════╦═════════════╗")
    print("║ # ║ Corr.  ║ Corr.  ║ Gaussiana   ║ Binomiale   ║")
    print("║   ║ Iniz.  ║ Stim.  ║   (µ, 𝜎)    ║ (n, prob)   ║")
    print("╠═══╬════════╬════════╬═════════════╬═════════════╣")

    for i in range(len(M)):
        # Print with Pearson coefficient included
        print(f"║{i+1:<3}║ {matr[i][0,1]:>5.2f}  ║ {matrStim[i][0,1]:>6.3f} ║ {parGauX[i][0]:>5.2f},{parGauX[i][1]:>5.2f} ║{parBinX[i][0]:>6.2f},{parBinX[i][1]:>5.2f} ║ (X)")
        print(f"║   ║        ║        ║ {parGauY[i][0]:>5.2f},{parGauY[i][1]:>5.2f} ║{parBinY[i][0]:>6.2f},{parBinY[i][1]:>5.2f} ║ (Y)")
        print("║   ║        ║        ║             ║             ║")
    print("╚═══╩════════╩════════╩═════════════╩═════════════╝")
