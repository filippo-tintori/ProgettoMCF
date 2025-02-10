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
#                       galton                      #
#         file con le classi di simulazione         #
#                                                   #
#####################################################

# -*- coding: utf-8 -*-

#------------------------------------------
#    Aggiungo moduli aggiuntivi
#------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr, spearmanr

# -
# Moduli propri
# -

from util import binomiale
from util import gaussiana

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

##################
#  CLASSE MADRE  #
##################

class GaltonBase:
    """
    Classe base della macchina di Galton
    
    GaltonBase
    
    Parametri
    ----------
        nPassi:     numero di passi della macchina
        nPalle:     numero di palle da simulazione
    
    """
    def __init__(self, nPassi, nPalle):
        self.nPassi = nPassi
        self.nPalle = nPalle



###################
#  CLASSI FIGLIE  #
###################


#---------------------------
#    Macchina di Galton 2D
#---------------------------

class Galton2D(GaltonBase):
    """
    Classe per la simulazione della macchina di Galton 2D
    
        Galton2D(GaltonBase)
    
    Parametri
    ---------
        nPassi, nPalle: derivati dalla classe GaltonBase
        probDX:         probabilità della palla di andare a destra 
        passiDX:        array dei bins con frequenze
    
    Metodi
    ------
        simula2D()          simula e restituisce il numero di passi verso destra di ogni pallina
        mostra2D(stampa)    visualizzazione 2D dei passi verso destra effettuati
            se stampa==True verranno mostrati i risultati dei fit

    """
    
    def __init__(self, nPassi, nPalle, probDX, stampa=True):
        """
        Costruttore della classe Galton2D.
        Crea un'istanza della classe e inizializza l'oggetto
        
        """
        super().__init__(nPassi, nPalle)
        self.probDX = probDX
        self.passiDX = None
        
        self.parBin = None
        self.parGau = None
        
        self.gdlBin = None
        self.gdlGau = None
        
        self.chi2Bin = None
        self.chi2Gau = None
        
        self.chi2RidBin = None
        self.chi2RidGau = None
        
        
        self.simula2D(stampa)
        self.mostra2D(stampa)
        
        
    def simula2D(self, stampa):
        """
        Simula il numero di passi verso destra che ogni palline effettua e
        li restituisce in un array
        
        """
        pDX = np.random.rand(self.nPalle, self.nPassi) < self.probDX
        self.passiDX = np.sum(pDX, axis = 1) # somma lungo il primo asse -> dim di self.nPalle
        
        if stampa:
            self.mostraTraiettoria2D(pDX)

        
    def mostra2D(self, stampa=True):
        """
        Mostra la distribuzione dei passi verso destro effettuati da ogni pallina
        con sovrapposto il fit della distribuzione binomiale e gaussiana
        
        """
        
        bordiBins = np.arange(-0.5, self.nPassi + 1, 1)
        istogr, Bordi = np.histogram(self.passiDX, bins=bordiBins,)
        centriBins = (bordiBins[:-1] + bordiBins[1:]) / 2
        
        
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))
        
        ############
        # SENZA FIT
        ############
        
        axs[0].bar(centriBins, istogr, width=1, color='skyblue', edgecolor='black', alpha=0.7, label=f"Osservata\n$n_{{\\mathrm{{PALLE}}}}$={self.nPalle}")
        axs[0].set_title("Senza fit")
        axs[0].set_xlabel("Passi verso DX")
        axs[0].set_ylabel("Frequenza")
        axs[0].legend()

        ############
        #  CON FIT
        ############

        axs[1].bar(centriBins, istogr, width=1, color='skyblue', edgecolor='black', alpha=0.7, label=f"Osservata\n$n_{{\\mathrm{{PALLE}}}}$={self.nPalle}")
        
        
        
        #-----------------------------
        # BINOMIALE tramite curve_fit
        #-----------------------------
        
        
        self.parBin, _ = curve_fit(binomiale, centriBins, istogr, 
                                   p0=[self.nPassi, self.probDX, np.max(istogr)], 
                                   absolute_sigma=True)
        
        xFit = np.linspace(min(centriBins), max(centriBins), 1000)
        
        fitBinom = binomiale(xFit, self.parBin[0], self.parBin[1], self.parBin[2])
        
        
        axs[1].plot(xFit, fitBinom, 'orange', label=f"Fit Binomiale\n$n={self.parBin[0]:.1f}, p={self.parBin[1]:.2f}$")
        
        binVal = binomiale(centriBins, self.parBin[0], self.parBin[1], self.parBin[2])
        
        maskBin = binVal > 0
        
        self.chi2Bin = np.sum((istogr[maskBin] - binVal[maskBin]) ** 2 / binVal[maskBin])
        self.gdlBin = np.sum(maskBin) - 3 # 3 parametri da ottimizzare: n, prob, amp
        
        self.chi2RidBin = self.chi2Bin / self.gdlBin
        
        
        #-----------------------------
        # GAUSSIANA tramite curve_fit
        #-----------------------------
        
        self.parGau, _ = curve_fit(gaussiana, centriBins, istogr, 
                                   p0=[np.mean(self.passiDX), np.std(self.passiDX), np.max(istogr)], 
                                   absolute_sigma=True)
        
        
        fitGauss = gaussiana(xFit, self.parGau[0], self.parGau[1], self.parGau[2])
        
        axs[1].plot(xFit, fitGauss, 'blue', label= f"Fit Gaussiana\n$\\mu={self.parGau[0]:.2f}, \\sigma={self.parGau[1]:.2f}$")
        
        gauVal = gaussiana(centriBins, self.parGau[0], self.parGau[1], self.parGau[2])
        
        maskGau = gauVal > 0

        self.chi2Gau = np.sum((istogr[maskGau] - gauVal[maskGau]) ** 2 / gauVal[maskGau])
        self.gdlGau = np.sum(maskGau) - 3  # 3 parametri da ottimizzare: media, sigma, amp
        
        self.chi2RidGau = self.chi2Gau / self.gdlGau

        axs[1].set_title("Con fit")
        axs[1].set_xlabel("Passi verso DX")
        axs[1].set_ylabel("Frequenza")
        axs[1].legend()
        
        fig.suptitle(f"Distribuzione dei passi verso DX della Macchina di Galton 2D\nnPassi={self.nPassi},  nPalle={self.nPalle},  probDX={self.probDX}")
        plt.tight_layout()
        
        
        if stampa:
            print("Parametri teorici:")
            print("  Binomiale:")
            print(f"    n={self.nPassi}")
            print(f"    p={self.probDX}")
            print("  Gaussiana:")
            print(f"    µ={(self.nPassi*self.probDX):.1f}")
            print(f"    𝜎={(np.sqrt(self.nPassi*self.probDX*(1-self.probDX))):.2f}")

            print("\nRisultati dei fit:")
            print("  Binomiale:")
            print(f"    n = {self.parBin[0]:.2f}")
            print(f"    p = {self.parBin[1]:.3f}\n")
            print(f"    GdL = {self.gdlBin}")
            print(f"    χ² = {self.chi2Bin:.2f}")
            print(f"    χ² rid = {self.chi2RidBin:.2f}")

            print("  Gaussiana:")
            print(f"    µ = {self.parGau[0]:.2f}")
            print(f"    𝜎 = {self.parGau[1]:.2f}\n")
            print(f"    GdL = {self.gdlGau}")
            print(f"    χ² = {self.chi2Gau:.2f}")
            print(f"    χ² rid = {self.chi2RidGau:.2f}")
            print("\n---")
            plt.show()
        else:
            plt.draw()
            plt.close()
    
    
    def mostraTraiettoria2D(self, passiX):
        """
        Visualizzazione 2D delle traiettorie delle palline lungo l'asse X.
        
        """
        passiX = np.where(passiX, 1, -1)
        
        traietX = np.zeros((self.nPalle, self.nPassi))
        
        for i in range(1,self.nPassi):
            traietX[:, i] = traietX[:, i-1] + passiX[:, i]
        
        plt.figure(figsize=(10, 7))
        plt.title(f'Traiettorie delle palline nella macchina di Galton 2D\nnPassi={self.nPassi},  nPalle={self.nPalle},  probDX={self.probDX}')
        plt.xlabel('Passi lungo X')
        plt.ylabel('nPassi (pioli)')
        
        for i in range(self.nPalle):
            plt.plot(traietX[i], np.arange(self.nPassi), alpha=0.6)
        
        plt.show()
    


#---------------------------
#    Macchina di Galton 3D
#---------------------------

class Galton3D(GaltonBase):
    """
    Classe per la simulazione della macchina di Galton 3D.
    
    Galton3D(GaltonBase)
    
    Parametri
    ---------
        nPassi, nPalle: derivati dalla classe GaltonBase
        probX:          probabilità della pallina di andare a destra lungo l'asse X
        probY:          probabilità della pallina di andare a destra lungo l'asse Y
        risultatiX:     array di passi simulati verso destra da ogni palline lungo X
        risultatiY:     array di passi simulati verso destra da ogni palline lungo Y
    
    
    Metodi
    ------
        simula3D(stampa)        simula il numero di passi verso destra di ogni pallina
                                    Se stampa==True fa vedere la traiettoria
        mostra3D()              visualizzazione 3D dei passi verso destra effettuati
        proiezioni3D(stampa)    visualizzazione 2D dei passi verso destra effettuati lungo X e lungo Y
        mostraMatrice()         visualizzazione di una matrice di frequenze delle palline nei bin
        
    """
    
    def __init__(self, nPassi, nPalle, probX, probY, stampa=True):
        """
        Costruttore della classe Galton3D.
        Crea un'istanza della classe e inizializza l'oggetto
        
        """
        super().__init__(nPassi, nPalle)
        self.probX = probX
        self.probY = probY
        self.risultatiX, self.risultatiY = None, None
        
        self.parBinX = None
        self.parBinY = None
        
        self.parGauX = None
        self.parGauY = None
        
        self.gdlBinX = None
        self.gdlBinY = None
        
        self.gdlGauX = None
        self.gdlGauY = None
        
        self.chi2BinX = None
        self.chi2BinY = None
        
        self.chi2GauX = None
        self.chi2GauY = None
        
        self.chi2RidBinX = None
        self.chi2RifBinY = None
        
        self.chi2RidGauX = None
        self.chi2RidGauY = None
        
        
        
        if type(self) == Galton3D:
            print("Avviata la simulazione della macchina di Galton 3D\n\n")
            
            self.simula3D(stampa)
            if stampa:
                self.mostra3D()
                self.mostraMatrice()
            self.proiezioni3D(stampa)

    
    def simula3D(self, stampa):
        """
        Simula il numero di passi verso destra che ogni palline effettua lungo l'asse X e Y
        in base alla probabilità probX e probY di andare verso destra.
        
        """
        # array casuali [nPalle, nPassi] con valori tra 0 e 1
        passiX = np.random.rand(self.nPalle, self.nPassi) < self.probX
        passiY = np.random.rand(self.nPalle, self.nPassi) < self.probY
        
        # somma di tutti i passi in X e Y
        self.risultatiX = np.sum(passiX, axis=1)
        self.risultatiY = np.sum(passiY, axis=1)
        
        if stampa:
            self.mostraTraiettoria3D(passiX, passiY)
        
        
    
    
    def mostra3D(self):
        """
        Mostra la rappresentazione 3D della macchina di Galton 3D.
        
        """
        #----------------
        # PLOT SENZA FIT
        #----------------
        
        # bordi dei bin
        bordiBin = np.arange(-0.5, self.nPassi + 1, 1)

        # dimensione istogramma bidimensionale
        istogramma, xBordi, yBordi = np.histogram2d(self.risultatiX, self.risultatiY, bins=[bordiBin, bordiBin])

        xPos, yPos = np.meshgrid(xBordi[:-1], yBordi[:-1], indexing="ij")  # 2 griglie 2D
        xPos = xPos.ravel()  # rende a 1D
        yPos = yPos.ravel()
        zPos = np.zeros_like(xPos)

        dx = dy = 1  # Larghezza bin
        dz = istogramma.ravel()  # frequenze dell'istogramma

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        
        ax.bar3d(xPos, yPos, zPos, dx, dy, dz, color='skyblue', edgecolor='royalblue', alpha=0.5)
        
        ax.set_title(f"Distribuzione 3D della Macchina di Galton\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}")
        ax.set_xlabel("Passi verso DX in X")
        ax.set_ylabel("Passi verso DX in Y")
        ax.set_zlabel("Frequenza")
        plt.show()
        
        #--------------
        # PLOT CON FIT
        #--------------
        
        # bordi dei bin
        bordiBin = np.arange(-0.5, self.nPassi + 1, 1)

        # dimensione istogramma bidimensionale
        istogramma, xBordi, yBordi = np.histogram2d(self.risultatiX, self.risultatiY, bins=[bordiBin, bordiBin])

        xPos, yPos = np.meshgrid(xBordi[:-1], yBordi[:-1], indexing="ij")  # 2 griglie 2D
        xPos = xPos.ravel()  # rende a 1D
        yPos = yPos.ravel()
        zPos = np.zeros_like(xPos)

        dx = dy = 1  # Larghezza bin
        dz = istogramma.ravel()  # frequenze dell'istogramma

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        
        ax.bar3d(xPos, yPos, zPos, dx, dy, dz, color='skyblue', edgecolor='royalblue', alpha=0.5)
        
        
        # Media e deviazione standard per X e Y
        muX, sigmaX = self.nPassi * self.probX, np.sqrt(self.nPassi * self.probX * (1 - self.probX))
        muY, sigmaY = self.nPassi * self.probY, np.sqrt(self.nPassi * self.probY * (1 - self.probY))
        
        # Griglia per il fit
        X = np.linspace(0, self.nPassi, 100)
        Y = np.linspace(0, self.nPassi, 100)
        X, Y = np.meshgrid(X, Y)
        
        # Distribuzione teorica
        Z = (norm.pdf(X, muX, sigmaX) * norm.pdf(Y, muY, sigmaY)) * np.sum(istogramma)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='r', alpha=0.8, edgecolor='none')

        ax.set_title(f"Distribuzione 3D della Macchina di Galton con fit\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}")
        ax.set_xlabel("Passi verso DX in X")
        ax.set_ylabel("Passi verso DX in Y")
        ax.set_zlabel("Frequenza")
        plt.tight_layout()
        plt.show()
    
    
    def proiezioni3D(self, stampa):
        """
        Funzione per calcolare e visualizzare le due proiezioni 2D.
        
            proiezioni3D(stampa=True)

        Se stampa==True, stamperà i fit delle distribuzioni
        
        """
    
        
        # Proiezione X
        proiezX, bordiX = np.histogram(self.risultatiX, bins=np.arange(-0.5, self.nPassi + 1, 1))
        centriX = (bordiX[:-1] + bordiX[1:]) / 2
        
        # Proiezione Y
        proiezY, bordiY = np.histogram(self.risultatiY, bins=np.arange(-0.5, self.nPassi + 1, 1))
        centriY = (bordiY[:-1] + bordiY[1:]) / 2
        
        
        # BINOMIALE tramite curve_fit (X e Y)
        
        # X
        self.parBinX, _ = curve_fit(binomiale, centriX, proiezX, 
                                    p0=[self.nPassi, self.probX, np.max(proiezX)],
                                    absolute_sigma=True)
        
        
        x_fit = np.linspace(min(centriX), max(centriX), 1000)
        fitBinomX = binomiale(x_fit, self.parBinX[0], self.parBinX[1], self.parBinX[2])
        
        
        
        
        binVal = binomiale(centriX, self.parBinX[0], self.parBinX[1], self.parBinX[2])
        maskBx = binVal > 0
        
        self.chi2BinX = np.sum((proiezX[maskBx] - binVal[maskBx]) ** 2 / binVal[maskBx])
        self.gdlBinX = np.sum(maskBx) - 3 # 3 parametri da ottimizzare: prob e ampiezza
        self.chi2RidBinX = self.chi2BinX / self.gdlBinX
        
        # Y
        self.parBinY, _ = curve_fit(binomiale, centriY, proiezY, 
                                    p0 = [self.nPassi, self.probY, np.max(proiezY)],
                                    absolute_sigma=True)
        
        
        y_fit = np.linspace(min(centriY), max(centriY), 1000)
        fitBinomY = binomiale(y_fit, self.parBinY[0], self.parBinY[1], self.parBinY[2])
        
        
        binValY = binomiale(centriY, self.parBinY[0], self.parBinY[1], self.parBinY[2])
        maskBy = binValY > 0
        
        self.chi2BinY = np.sum((proiezY[maskBy] - binValY[maskBy]) ** 2 / binValY[maskBy])
        self.gdlBinY = np.sum(maskBy) - 3
        self.chi2RidBinY = self.chi2BinY / self.gdlBinY
        
        
        # GAUSSIANA tramite curve_fit (X e Y)
        
        # X
        
        self.parGauX, _ = curve_fit(gaussiana, centriX, proiezX, 
                                    p0=[np.mean(self.risultatiX), np.std(self.risultatiX), np.max(proiezX)],
                                    absolute_sigma=True)
        
        
        fitGaussX = gaussiana(x_fit, self.parGauX[0], self.parGauX[1], self.parGauX[2])
        
        
        
        gauVal = gaussiana(centriX, self.parGauX[0], self.parGauX[1], self.parGauX[2])
        
        maskGx = gauVal > 0
        
        self.chi2GauX = np.sum((proiezX[maskGx] - gauVal[maskGx]) ** 2 / gauVal[maskGx])
        self.gdlGauX = np.sum(maskGx) - 3  # 3 parametri da ottimizzare: media, sigma, amp
        self.chi2RidGauX = self.chi2GauX / self.gdlGauX
        
        # Y
        
        self.parGauY, _ = curve_fit(gaussiana, centriY, proiezY, 
                                    p0=[np.mean(self.risultatiY), np.std(self.risultatiY), np.max(proiezY)],
                                    absolute_sigma=True)
        
        
        fitGaussY = gaussiana(y_fit, self.parGauY[0], self.parGauY[1], self.parGauY[2])
        
        
        
        gauValY = gaussiana(centriY, self.parGauY[0], self.parGauY[1], self.parGauY[2])
        
        maskGy = gauValY > 0
        
        self.chi2GauY = np.sum((proiezY[maskGy] - gauValY[maskGy]) ** 2 / gauValY[maskGy])
        self.gdlGauY = np.sum(maskGy) - 3
        self.chi2RidGauY = self.chi2GauY / self.gdlGauY
    
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharey=True) # y condivisa

        # Grafico 1: Proiezione X
        ax[0, 0].bar(centriX, proiezX, width=1, color='skyblue', edgecolor='royalblue', alpha=0.7, label="Osservata")
        ax[0, 0].set_title("Proiezione X")
        ax[0, 0].set_xlabel("Passi in X")
        ax[0, 0].set_ylabel("Frequenza")
        ax[0, 0].legend()

        # Grafico 2: Proiezione Y
        ax[0, 1].bar(centriY, proiezY, width=1, color='lightgreen', edgecolor='green', alpha=0.7, label="Osservata")
        ax[0, 1].set_title("Proiezione Y")
        ax[0, 1].set_xlabel("Passi in Y")
        ax[0, 1].legend()

        # Grafico 3: Fit X
        ax[1, 0].bar(centriX, proiezX, width=1, color='skyblue', edgecolor='royalblue', alpha=0.7, label="Osservata")
        ax[1, 0].plot(x_fit, fitBinomX, 'orange', label=f"Binomiale\n$n={self.parBinX[0]:.1f}, p={self.parBinX[1]:.2f}$")
        ax[1, 0].plot(x_fit, fitGaussX, 'blue', label=f"Guassiana\n$\\mu={self.parGauX[0]:.2f}, \\sigma={self.parGauX[1]:.2f}$")
        ax[1, 0].set_title("Proiezione X con fit")
        ax[1, 0].set_xlabel("Passi in X")
        ax[1, 0].set_ylabel("Frequenza")
        ax[1, 0].legend()

        # Grafico 4: Fit Y
        ax[1, 1].bar(centriY, proiezY, width=1, color='lightgreen', edgecolor='green', alpha=0.7, label="Osservata")
        ax[1, 1].plot(y_fit, fitBinomY, 'orange', label=f"Binomiale\n$n={self.parBinY[0]:.1f}, p={self.parBinY[1]:.2f}$")
        ax[1, 1].plot(y_fit, fitGaussY, 'blue', label=f"Gaussiana\n$\\mu={self.parGauY[0]:.2f}, \\sigma={self.parGauY[1]:.2f}$")
        ax[1, 1].set_title("Proiezione Y con fit")
        ax[1, 1].set_xlabel("Passi in Y")
        ax[1, 1].legend()

        plt.suptitle(f"Distribuzione delle Proiezioni\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}")
        plt.tight_layout()
        plt.show()
        
        if stampa:
            print("\n---\n")
            print("Parametri teorici:")
            print("  Binomiale:")
            print("             X       |       Y")
            print("    n:      {:<6}   |    {:<6}".format(self.nPassi, self.nPassi))
            print("    p:      {:<6.3f}   |    {:<6.3f}".format(self.probX, self.probY))

            print("  Gaussiana:")
            print("             X       |       Y")
            print("    µ:      {:<6.1f}   |    {:<6.1f}".format(self.nPassi * self.probX, self.nPassi * self.probY))
            print("    𝜎:      {:<6.2f}   |    {:<6.2f}".format(
                np.sqrt(self.nPassi * self.probX * (1 - self.probX)),
                np.sqrt(self.nPassi * self.probY * (1 - self.probY))
            ))
    
            print("\n\nRisultati dei fit:")
            print("  Binomiale:")
            print("             X       |       Y")
            print("    n:      {:<6.3f}   |    {:<6.3f}".format(self.parBinX[0], self.parBinY[0]))
            print("    p:      {:<6.3f}   |    {:<6.3f}".format(self.parBinX[1], self.parBinY[1]))
            print("    GdL:    {:<6}   |    {:<6}".format(self.gdlBinX, self.gdlBinY))
            print("    χ²:     {:<6.2f}   |    {:<6.2f}".format(self.chi2BinX, self.chi2BinY))
            print("    χ²rid:  {:<6.2f}   |    {:<6.2f}".format(self.chi2RidBinX, self.chi2RidBinY))
    
            print("\n  Gaussiana:")
            print("             X       |       Y")
            print("    µ:      {:<6.2f}   |    {:<6.2f}".format(self.parGauX[0], self.parGauY[0]))
            print("    𝜎:      {:<6.2f}   |    {:<6.2f}".format(self.parGauX[1], self.parGauY[1]))
            print("    GdL:    {:<6}   |    {:<6}".format(self.gdlGauX, self.gdlGauY))
            print("    χ²:     {:<6.2f}   |    {:<6.2f}".format(self.chi2GauX, self.chi2GauY))
            print("    χ²rid:  {:<6.2f}   |    {:<6.2f}".format(self.chi2RidGauX, self.chi2RidGauY))
            print("\n---")
    
    
    
    def mostraMatrice(self):
        """
        Mostra una matrice di bin colorati in base alla frequenza di ognuno.
        
        """

        bordiBin = np.arange(-0.5, self.nPassi + 1, 1)  # Bin centrati su interi
        istogramma, xBordi, yBordi = np.histogram2d(self.risultatiX, self.risultatiY, bins=[bordiBin, bordiBin])

        frequenze = istogramma.T  # Trasposta per allineare agli assi X e Y

        plt.figure(figsize=(8, 6))
        plt.imshow(frequenze, origin='lower', cmap='Blues', extent=[-0.5, self.nPassi + 0.5, -0.5, self.nPassi + 0.5])
        plt.colorbar(label='Frequenza')  # Barra dei colori


        plt.title(f"Matrice dei risultati della Macchina di Galton 3D\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}")
        plt.xlabel("Passi verso DX in X")
        plt.ylabel("Passi verso DX in Y")

        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()
    
    def mostraTraiettoria3D(self, passiX, passiY):
        """
        Visualizzazione 3D delle traiettorie delle palline lungo gli assi X e Y.
        
        """
        passiX = np.where(passiX, 1, -1)
        passiY = np.where(passiY, 1, -1)
        
        traietX = np.zeros((self.nPalle, self.nPassi))
        traietY = np.zeros((self.nPalle, self.nPassi))
        
        for i in range(1,self.nPassi):
            traietX[:, i] = traietX[:, i-1] + passiX[:, i]
            traietY[:, i] = traietY[:, i-1] + passiY[:, i]
            
        
        
        plt.figure(figsize=(10, 7))
        ax = plt.axes(projection='3d')
        ax.set_title(f'Traiettorie delle palline\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}')
        ax.set_xlabel('Passi lungo X')
        ax.set_ylabel('Passi lungo Y')
        ax.set_zlabel('nPassi (pioli)')
        
        for i in range(self.nPalle):
            ax.plot(traietX[i], traietY[i], np.arange(self.nPassi), alpha=0.6)
        
        plt.show()


    

#-------------------------------------
#    Macchina di Galton 3D correlata
#-------------------------------------

class Galton3Dcorr(Galton3D):
    """
    Classe per la simulazione della macchina di Galton 3D con correlazione tra le gli assi.
    
    Galton3Dcorr(Galton3D)
    
    Parametri
    ---------
        nPassi, nPalle:             derivati dalla classe GaltonBase
        probX, probY:               derivati dalla classe Galton3D
        risultatiX, risultatiY:     derivati dalla classe Galton3D
        matriceCorrelazione:        matrice di correlazione in input
        matriceCorrStimata:         matrice di correlazione stimata dai risultatiX,Y
    
    Metodi
    ------
        simula3D()              simula il numero di passi verso destra di ogni pallina
        mostra3D()              visualizzazione 3D dei passi verso destra effettuati    
        proiezioni3D(stampa)    derivato dalla classe Galton3D
        mostraMatrice()         derivato dalla classe Galton3D
    
    """
    
    def __init__(self, nPassi, nPalle, probX, probY, matriceCorrelazione, stampa=True):
        """
        Costruttore della classe Galton3Dcorr.
        Crea un'istanza della classe e inizializza l'oggetto
        
        """
       
        super().__init__(nPassi, nPalle, probX, probY)
        
        self.matriceCorrelazione = matriceCorrelazione
        self.matriceCorrStimata = None
        
        print("\n----\n")
        print("Avviata la simulazione della macchina di Galton 3D con correlazione\n\n")
        
        self.simula3D(stampa)
        
        if stampa:
            self.mostra3D()
        self.mostraMatrice()
        self.proiezioni3D(stampa)
        
        

    def simula3D(self, stampa):
        """
        Simula la macchina di Galton 3D influenzata dalla correlazione tra gli assi X e Y
        dalla matrice di correlazione.
        
        Calcola la matrice di correlazione stimata dai passi simulati
        
        Calcola il coefficiente di correlazione di Pearson
        
        """
        
        # decomposizione di Cholesky -> matrice triangolare inferiore
        L = np.linalg.cholesky(self.matriceCorrelazione)
        
        
        passi_indipendenti = np.random.randn(self.nPalle, 2, self.nPassi)  # forma (nPalle, 2, nPassi)

        # Applicazione della correlazione tramite la decomposizione di Cholesky
        passi_correlati = np.einsum('ij,njp->nip', L, passi_indipendenti)  # forma (nPalle, 2, nPassi)

        # suddivido
        passiX = passi_correlati[:, 0, :] < self.probX   # forma (nPalle, nPassi)
        passiY = passi_correlati[:, 1, :] < self.probY

        
        self.risultatiX = np.sum(passiX, axis=1)
        self.risultatiY = np.sum(passiY, axis=1)
        
        
        passi_aggregati = passi_correlati.sum(axis=2)  # forma (nPalle, 2) - aggrega sui passi
        
        self.matriceCorrelazioneStimata = np.corrcoef(passi_aggregati.T)  # matrice stimata
        

        print("=" * 45)
        print(" MATRICE DI CORRELAZIONE INSERITA ")
        print("=" * 45)
        print("\n".join(["[" + "  ".join([f"{val:.2f}" for val in row]) + "]" for row in self.matriceCorrelazione]))
        print("\n")

        print("=" * 45)
        print(" MATRICE DI CORRELAZIONE STIMATA DAI PASSI ")
        print("=" * 45)
        print("\n".join(["[" + "  ".join([f"{val:.3f}" for val in row]) + "]" for row in self.matriceCorrelazioneStimata]))
        print("=" * 45)
        
        if stampa:
            self.mostraTraiettoria3D(passiX, passiY)
        
        
        
    def mostra3D(self):
        """
        Visualizza la distribuzione 3D delle palline come istogramma.
        
        """
        
        #----------------
        # PLOT SENZA FIT
        #----------------
        
        bordiBin = np.arange(-0.5, self.nPassi + 1, 1)

        
        istogramma, xBordi, yBordi = np.histogram2d(self.risultatiX, self.risultatiY, bins=[bordiBin, bordiBin])

        
        xPos, yPos = np.meshgrid(xBordi[:-1], yBordi[:-1], indexing="ij")
        xPos = xPos.ravel()
        yPos = yPos.ravel()
        zPos = np.zeros_like(xPos)

        dx = dy = 1  # larghezza dei bin
        dz = istogramma.ravel()  # frequenze dell'istogramma

        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # istogramma 3D 
        ax.bar3d(xPos, yPos, zPos, dx, dy, dz, color='skyblue', edgecolor='royalblue', alpha=0.5, label='Istogramma Simulato')

        ax.set_title(f"Distribuzione 3D della Macchina di Galton\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}")
        ax.set_xlabel("Passi verso DX in X")
        ax.set_ylabel("Passi verso DX in Y")
        ax.set_zlabel("Frequenza")
        ax.legend()

        plt.show()
        
        
        #--------------
        # PLOT CON FIT
        #--------------

        bordiBin = np.arange(-0.5, self.nPassi + 1, 1)

        istogramma, xBordi, yBordi = np.histogram2d(self.risultatiX, self.risultatiY, bins=[bordiBin, bordiBin])

        
        xPos, yPos = np.meshgrid(xBordi[:-1], yBordi[:-1], indexing="ij")
        xPos = xPos.ravel()
        yPos = yPos.ravel()
        zPos = np.zeros_like(xPos)

        dx = dy = 1  # larghezza dei bin
        dz = istogramma.ravel()  # frequenze dell'istogramma

        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # istogramma 3D 
        ax.bar3d(xPos, yPos, zPos, dx, dy, dz, color='skyblue', edgecolor='royalblue', alpha=0.5, label='Istogramma Simulato')
            
        x_range = np.arange(-0.5, self.nPassi + 1, 0.5)
        y_range = np.arange(-0.5, self.nPassi + 1, 0.5)
        X, Y = np.meshgrid(x_range, y_range)
        
        
        mean = [np.mean(self.risultatiX), np.mean(self.risultatiY)]
        stdX = np.std(self.risultatiX)
        stdY = np.std(self.risultatiY)
        cov = self.matriceCorrelazione * (stdX * stdY)  # matrice di covarianza
        
        # distribuzione normale multivariata
        rv = multivariate_normal(mean, cov)
        Z = rv.pdf(np.dstack((X, Y))) * self.nPalle  # densità di probabilità scalata per il numero di palline
        
        # distribuzione attesa
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, label='Distribuzione Attesa')

        ax.set_title(f"Distribuzione 3D della Macchina di Galton con fit\nnPassi={self.nPassi},  nPalle={self.nPalle},  probX={self.probX},  probY={self.probY}")
        ax.set_xlabel("Passi verso DX in X")
        ax.set_ylabel("Passi verso DX in Y")
        ax.set_zlabel("Frequenza")
        ax.legend()
        plt.show()