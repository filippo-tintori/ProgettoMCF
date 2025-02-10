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
#                        MAIN                       #
#                                                   #
#####################################################

# -*- coding: utf-8 -*-

#------------------------------------------
#    Aggiungo moduli aggiuntivi
#------------------------------------------

import argparse
import numpy as np

# -
# Moduli propri
# -

from galton import Galton2D
from galton import Galton3D
from galton import Galton3Dcorr

from util import NumPalAdeg
from util import verificaMatriCorr

from studio2D import studio2D
from studio3D import studio3D
from studio3Dcorr import studio3Dcorr



#--------------------------------------
# Descrizione della macchina di Galton
#--------------------------------------

descrizione = """Questo codice permette di simulare una macchina di Galton in 2D, 3D e 3D correlata.

Nel caso 2D, le palline hanno una probabilità p di andare a destra e 1-p di andare a sinistra ad ogni piolo.

Nel caso 3D, le palline hanno probabilità pX e pY di andare a destra lungo gli assi X e Y rispettivamente.

Nel caso 3D con correlazione, la condizione di caduta delle palline è condizionata in aggiunta 
da una matrice di correlazione 2x2 che determina la correlazione tra lo spostamento lungo l'asse X e 
lungo l'asse Y.
"""

generale = """La macchina di Galton è un dispositivo inventato da Sir Francis Galton per fornire una dimostrazione 
pratica del teorema del limite centrale e della distribuzione normale.

Una volta fatte cadere le palline da una fessura in alto, esse incontrano una serie di pioli disposti 
secondo la configurazione del quinconce. Ad ogni piolo, le palline possono cadere a destra o a sinistra 
con una probabilità fissata. Finite le cadute, le palline si raccolgono in una serie di contenitori.

Dopo aver simulato la caduta di un adeguato numero di palline, si verificherà che la distribuzione delle palline
sarà la distribuzione binomiale. Che per un numero adeguato di cadute, si avvicinerà alla distribuzione
gaussiana o normale.

Il valore medio della distribuzione Gaussiana sarà: X = n*p.
La deviazione standard della distribuzione Gaussiana sarà: sigma = sqrt(n*p*(1-p)).
n = numero di cadute.
p = probabilità di cadere a destra.
"""


#----------------------------------------------------
#    Funzione per gestire gli argomenti di ArgParse
#----------------------------------------------------

def parse_arguments():
    
    # gestione argomenti e opzioni
    parser = argparse.ArgumentParser(description=descrizione, epilog=generale,
                                     usage      ='python3 Progetto.py  --option')
    
    
    parser.add_argument('--dim2',          '-d2',               action='store_true',    help='Macchina di Galton 2D')
    parser.add_argument('--dim3',          '-d3',               action='store_true',    help='Macchina di Galton 3D')
    parser.add_argument('--dim3corr',     '-d3c',               action='store_true',    help='Macchina di Galton 3D con correlazione tra gli assi X e Y')
    parser.add_argument('--nPalle',        '-np',               type=int,               help='Numero di palline della macchina (nPalle > 1, default=numero adeguato)')
    parser.add_argument('--nPassi',         '-n',               type=int,               help='Numero di passi della macchina (nPassi > 0, default=10)')
    parser.add_argument('--probX',         '-px',               type=float,             help='Probabilità di andare a destra lungo X ( 0 <= probX <= 1)')
    parser.add_argument('--probY',         '-py',               type=float,             help='Probabilità di andare a destra lungo Y ( 0 <= probX <= 1)')
    parser.add_argument('--matrice',        '-m',     nargs=4,  type=float,             help='Inserisci la matrice di correlazione (es. 1 0.9 0.9 1)')
    parser.add_argument('--margineErrore', '-me',               type=float,             help='Margine di errore per il calcolo del numero di palline da simulare (es. 0.05 per il 5%%)')
    
    parser.add_argument('--studio2D',       '-s2d',             action='store_true',    help='Studio effettuato per la macchina di Galton 2D')
    parser.add_argument('--studio3D',       '-s3d',             action='store_true',    help='Studio effettuato per la macchina di Galton 3D')
    parser.add_argument('--studio3Dcorr',   '-s3dc',            action='store_true',    help='Studio effettuato per la macchina di Galton 3D correlata')
    
    args = parser.parse_args()
    
    
    #------------------------------
    # controllo argomenti inseriti
    #------------------------------
    
    print("\n")
    # tipo di macchina
    if sum([args.dim2, args.dim3, args.dim3corr, args.studio2D, args.studio3D, args.studio3Dcorr]) != 1:
        parser.error("Devi selezionare uno e uno solo tra --dim2, --dim3, --dim3corr --studio2d, --studio3D e --studio3Dcorr.")
    
    if sum([args.studio2D, args.studio3D, args.studio3Dcorr]) == 0:
        if (args.margineErrore is None and args.nPalle is None):
            parser.error("Devi inserire un parametro tra --margineErrore e --nPalle.")

        if args.margineErrore is not None and args.nPalle is not None:
            print("Hai inserito sia --margineErrore che --nPalle, verrà data preferenza a quest'ultimo.")
            
        if args.margineErrore is not None and (args.margineErrore <= 0.0 or args.margineErrore >= 1.0):
                    parser.error("Il margine di errore --margineErrore deve essere compreso tra 0 e 1.")

        
        if args.probX is None:
            parser.error("Devi fornire la probabilità --probX per la simulazione.")

        if args.probX <= 0 or args.probX >= 1:
            parser.error("La probabilità --probX deve essere compresa tra 0 e 1 (esclusi).")

        if args.nPassi is None:
            parser.error("Devi fornire il numero di passi --nPassi")

        if args.nPassi < 1:
            parser.error("Il numero di passi della macchina deve essere maggiore di 0.")    

        if args.nPalle is None:
            args.nPalle = NumPalAdeg(args.nPassi, args.probX, args.margineErrore, args.probY)

        if args.nPalle < 1:
            parser.error("Devi fornire un numero di palline --nPalle che sia maggiore di 0.")


        if args.dim3:
            if args.probY is None:
                parser.error("Devi fornire sia --probX che --probY per la simulazione 3D.")
            if args.probY <= 0 or args.probY >= 1:
                parser.error("La probabilità --probY deve essere compresa tra 0 e 1 (esclusi).")


        if args.dim3corr:
            if args.probY is None:
                parser.error("Devi fornire sia --probX che --probY per la simulazione 3D correlata.")
            elif args.probY <= 0 or args.probY >= 1:
                parser.error("La probabilità --probY deve essere compresa tra 0 e 1 (esclusi).")
            if args.matrice is None:
                parser.error("Devi fornire la matrice di correlazione --matrice per la simulazione 3D correlata.")
            args.matrice = np.array(args.matrice).reshape(2,2)
            if not verificaMatriCorr(args.matrice):
                return parser.error("La matrice di correlazione non è valida.")




        if args.probY is not None and (args.probY < 0 or args.probY > 1):
            parser.error("La probabilità --probY deve essere compresa tra 0 e 1.")

        print("Numero di palle utilizzato:",args.nPalle)

    print("\nArgomenti inseriti controllati e utilizzabili.\n")
    
    
    print("\n")
    print("----------------------")
    print("  MACCHINA DI GALTON  ")
    print("----------------------")
    print("\nArgomenti inseriti:\n")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    print("\n----------------------\n\n")

    
    return  args




#----------------------------------
#    Funzione per gestire ArgParse
#----------------------------------


def argsFunzione(args):
    """
    Gestisce gli argomenti in input dell'utente
    
    Parametri
    ---------
        args:   argomenti di ArgParse
    
    """
    
    #-------------------------------------------
    #  Simulazione della macchina di Galton 2D
    #-------------------------------------------
    
    if args.dim2:
        Galton2D(args.nPassi, args.nPalle, args.probX)
    
    
    #-------------------------------------------
    #  Simulazione della macchina di Galton 3D
    #-------------------------------------------
    
    if args.dim3:
        Galton3D(args.nPassi, args.nPalle, args.probX, args.probY)
    
    
    #-----------------------------------------------------
    #  Simulazione della macchina di Galton 3D correlata 
    #-----------------------------------------------------
    
    if args.dim3corr:
        Galton3Dcorr(args.nPassi, args.nPalle, args.probX, args.probY, args.matrice) 
    
    
    
    
    #--------------------------------------#
    #  Studio per le macchine 2D           #
    #--------------------------------------#
    
    if args.studio2D:
        studio2D()
    
    
    #--------------------------------------
    # Studio per le macchine 3D 
    #--------------------------------------
    
    
    if args.studio3D:
        studio3D()
    
    
    #--------------------------------------
    # Studio per le macchine 3D correlata
    #--------------------------------------
    
    if args.studio3Dcorr:
        studio3Dcorr()

    
#------------#
#    MAIN    #
#------------#


# ---
# Definizione della funzione di main
# ---

def main():
    args = parse_arguments()
    
    argsFunzione(args)


# ---
# Blocco di esecuzione principale o Entry point
# ---

if __name__ == "__main__":
    main()