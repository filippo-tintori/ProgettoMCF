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
#                        util                       #
#     file con le funzioni generali di utilità      #
#                                                   #
#####################################################


#--------------------------------
#    Aggiungo moduli aggiuntivi
#--------------------------------

import numpy as np
from scipy.special import comb



#----------------------------------------------
#    Funzioni per la Distribuzione Binomiale
#----------------------------------------------

def binomiale(x, n, p, amp):
    """
    Funzione per calcolare la distribuzione binomiale amplificata.
    Prova con il metodo esatto di "comb" e se fallisce usa il metodo
    approssimato.
    
    Parametri
    ---------
        x (int):     Numero di successi osservati.
        n (int):     Numero totale di prove.
        p (float):   Probabilità di successo in una singola prova.
        amp (float): Fattore di amplificazione per la distribuzione.
    
    Restituisce:
        float:   Valore della distribuzione binomiale normalizzata moltiplicata per il fattore di amplificazione.
    """
    
    try:
        b = amp * comb(n, x, exact=True) * (p ** x) * ((1 - p) ** (n - x))    # modalità esatta
    except:
        b = amp * comb(n, x, exact=False) * (p ** x) * ((1 - p) ** (n - x))   # approssimazione
    
    return b




#-----------------------------------------------
#    Funzioni per la Distribuzione Gaussiana
#-----------------------------------------------

def gaussiana(x, mu, sigma, amp):
    """
    Funzione per calcolare la distribuzione gaussiana amplificata.
    
        gaussiana(x, mu, sigma, amp)
    
    Parametri
    ---------
        x (float):      Il valore o i valori in cui calcolare la distribuzione.
        mu (float):     La media della distribuzione.
        sigma (float):  La deviazione standard della distribuzione.
        amp (float):    L'ampiezza della distribuzione.
        
        
    Restituisce:
        float:          Valore della distribuzione gaussiana normalizzata per il valore di X.
        
    """
    
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)




#--------------------------------------------
#    Gestione della matrice di correlazione 
#--------------------------------------------

def verificaMatriCorr(matrice):
    """
    Funzione che verifica se la matrice 2x2 in input è una matrice di correlazione valida.

    Parametri
    ---------
    matrice: numpy.ndarray (2x2)
        Matrice da verificare.
    
    
    Restituisce:
        bool:   True se e solo se la matrice inserita è una matrice di correlazione valida, 
                False altrimenti.

    """
    
    # simmetria
    if not np.allclose(matrice, matrice.T):
        print("Errore: La matrice NON è simmetrica.")
        return False
    
    # diagonale di 1
    if not np.allclose(np.diagonal(matrice), np.ones(matrice.shape[0])):
        print("Errore: La diagonale principale NON è composta da tutti 1.")
        return False
    
    # [-1, 1]
    if np.any((matrice < -1) | (matrice > 1)):
        print("Errore: Tutti i valori della matrice devono essere compresi tra -1 e 1.")
        return False
    
    #  semidefinita positiva - autovalori non negativi
    autovalori = np.linalg.eigvalsh(matrice)
    if np.any(autovalori < 0):
        print("Errore: La matrice NON è semidefinita positiva (ha autovalori negativi).")
        return False
    
    return True




#------------------------------------------
#    Stima del Numero di palline adeguate
#------------------------------------------

def NumPalAdeg(nPassi, probX, margineErrore, probY=None):
    """
    Calcola il numero di palline adeguato per una macchina di Galton 2D, 3D o 3D correlata.
    
        NumPalAdeg(nPassi, probX, margineErrore, probY=None)

    Parametri
    ---------
        nPassi (int): Numero di passi (pioli) della macchina.
        probX (float): Probabilità di andare a destra lungo l'asse X.
        margineErrore (float): Margine di errore desiderato (es. 0.05 per il 5%).
        probY (float, opzionale): Probabilità di andare a destra lungo l'asse Y (solo per 3D).


    Restituisce:
        int: Numero adeguato di palline.
        
    """
    
    if (probX==0 or probY==0) or (probX==1 or probY==1):
        return 1000000 # evito errore di bordi
    
    # Deviazione standard lungo X
    sigmaX = np.sqrt(nPassi * probX * (1 - probX))
    numeroPallineX = (sigmaX ** 2) / (margineErrore ** 2)

    if probY is not None:
        sigmaY = np.sqrt(nPassi * probY * (1 - probY))
        numeroPallineY = (sigmaY ** 2) / (margineErrore ** 2)
        return int(np.ceil(max(numeroPallineX, numeroPallineY)))
    
    return int(np.ceil(numeroPallineX))