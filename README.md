# Macchina di Galton

La macchina di Galton è un dispositivo progettato per illustrare il comportamento della distribuzione binomiale attraverso il movimento casuale di palline che cadono attraverso una griglia di pioli. Durante il percorso, ogni pallina rimbalza casualmente a sinistra o a destra, accumulandosi alla base in colonne che formano una distribuzione approssimativamente a campana, rappresentando visivamente il teorema centrale del limite. Nel programma viene simulato il funzionamento della macchina sia in due dimensioni, dove le traiettorie delle palline sono rappresentate su un piano, sia in tre dimensioni, offrendo una visualizzazione spaziale più immersiva che evidenzia il comportamento statistico delle palline in uno spazio tridimensionale.

È inoltre possibile scegliere una probabilità di caduta a destra che influenzerà il comportamento delle palline.

## Indice

* [Introduzione](#Introduzione)
* [File nella repository](#File-nella-repository)
* [Struttura del progetto](#Struttura-del-progetto)
* [Struttura del codice](#Struttura-del-codice)
* [Prerequisiti](#Prerequisiti)
    * [Installazione di Python](#Python)
    * [Installazione delle librerie di Python](#Librerie-di-Python)
    * [Installazione di Git](#Git)
    * [Clonazione della repository del progetto](#Clonazione-della-repository)

* [Note preliminari all'esecuzione](#Note-preliminari-all'esecuzione)
* [Esecuzione del programma](#Esecuzione-del-programma)
* [Simulazioni](#Simulazioni)
* [Studio delle simulazioni](##Studio-delle-simulazioni)
    * [Studio 2D](#Studio-2D)
    * [Studio 3D](#Studio-3D)
    * [Studio 3D correlato](#Studio-3D-correlato)
* [Spiegazione delle decisioni prese](#Spiegazione-delle-decisioni-prese)
    * [Spiegazione generale del codice](#Spiegazione-generale-del-codice)
    * [Numero di palline adeguato per la simulazione](#Numero-di-palline-adeguato-per-la-simulazione)
    * [Uso di scipy.special.comb](#Uso-di-scipy.special.comb)
    * [Simulazione dei passi dipendenti dalla correlazione](#Simulazione-dei-passi-dipendenti-dalla-correlazione)
    * [Calcolo della matrice di correlazione stimata](#Calcolo-della-matrice-di-correlazione-stimata)
    * [Calcolo del $\chi^2$](#Calcolo-del-$\chi^2$)
* [Referenze](#Referenze)
* [Licenza](#Licenza)

## Introduzione

La macchina di Galton è composta da una serie di pioli disposti a forma di quinconce su una superficie verticale. Una pallina viene fatta cadere dall'alto, rimbalza contro i pioli e, a ogni rimbalzo, può andare a destra o a sinistra con probabilità uguale o diverse. 

La probabilità di $\frac{1}{2}$ si ha quando la macchina non è inclinata e le palline non hanno una direzione dominante; se invece la macchina è inclinata, la pallina avrà una direzione preferita. L'inclinazione della macchina è una sorta di errore sistematico che si aggiunge alla macchina di Galton reale. 

Ogni pallina andrà a finire in un compartimento (bin) e dopo la caduta di un numero grande di palle, si potrà osservare la tendenza delle palline a preferire dei bin rispetto ad altri. Questo darà luogo ad una precisa distribuzione lungo gli assi disponibili.

La distribuzione delle palline segue una **distribuzione binomiale** se consideriamo il numero di passi (rimbalzi) come un numero finito. Tuttavia, sotto determinate condizioni, la distribuzione di Galton si avvicina alla distribuzione gaussiana.

$$
B_{n,p}(\nu) = P(\nu \text{ successi in } n \text{ prove}) = \binom{n}{\nu} p^\nu (1-p)^{n-\nu}
$$

Dove:

$$
\binom{n}{\nu} = \frac{n!}{\nu!(n-\nu)!}
$$

P($\nu$) è la probabilità che la palline finisca nel $\nu$-esimo contenitore della macchina di Galton.

Se si verificano le condizioni di approssimazione:
$$
B_{n,p} \approx G_{X,\sigma}(\nu) \ \ \ \ \ \ \ \ \ \ (n \  \text{grande})
$$
$$
\text{con} \ \ \ X = np \ \ \ \ \ \ \ \ \ \ \ \sigma = \sqrt{np(1-p)}
$$

Le condizioni di approssimazione sono le seguenti:
- il numero di numero di passi deve essere sufficientemente grande;
- il numero di palline deve essere adeguato;
- la probabilità di andare a DX deve essere vicina a $\frac{1}{2}$, probabilità per cui la distribuzione binomiale è simmetrica.

## File nella repository

In questa sezione vedremo il contenuto della repository Git con una spiegazione della loro utilità.

- ***main.py*** : gestisce l'esecuzione del codice, richiamando le funzioni in base all'input dell'utente.

- ***galton.py*** : contiene le classi utili a simulare le diverse macchine di Galton.

- ***util.py*** : contiene delle funzioni utili al programma.

- ***studio2D.py*** : contiene le funzioni utilizzate per lo studio delle simulazioni 2D.

- ***studio3D.py*** : contiene le funzioni utilizzate per lo studio delle simulazioni 3D.

- ***studio3Dcorr.py*** : contiene le funzioni utilizzate per lo studio delle simulazioni 3D correlate.

- ***verificaLibrerie.py*** :  utile a verificare la corretta installazione dei moduli Python utili all'esecuzione.

-  ***requirements.txt*** : utile a installare le librerie Python nel caso in cui non si abbiano.

- ***Macchina_di_Galton.pdf*** : documento con la descrizione del progetto da creare con le varie richieste.

- ***README.md*** : (questo file di testo) serve all'utente a comprendere come è stato organizzato il codice, cosa ci si può fare e come eseguirlo.

- ***LICENSE*** : descrive la licenza a cui è sottoposta la directory e di come l'utente può disporne.

La licenza utilizzata è quella ***MIT***.

## Struttura del progetto

Il progetto è organizzato come segue:

- `main.py`: File principale per l'esecuzione;
- `galton.py`: Contiene le classi che implementano le simulazioni;
- `util.py`: Funzioni generali di utilità;
- `studio2D.py`: Funzione per lo studio della simulazione 2D;
- `studio3D.py`: Funzione per lo studio della simulazione 3D;
- `studio3Dcorr.py`: Funzione per lo studio della simulazione 3D correlata.


## Struttura del codice

In questa sezione si mostrerà una rappresentazione grafica della struttura delle classi con riferimento ai relativi parametri e metodi.

Il codice è all'interno del file *galton.py*.

<br>

**GaltonBase** `(nPassi, nPalle)`

- **Galton2D** `(GaltonBase + probDX, _stampa=True)`  
  - `simula2D(self, stampa)`  
  - `mostra2D(self, stampa)`  
  - `mostraTraiettoria2D()` 
- **Galton3D** `(GaltonBase + probX, probY, _stampa=True)`  
  - `simula3D(self, stampa)`  
  - `mostra3D(self)`  
  - `proiezioni3D(self, stampa)`  
  - `mostraMatrice(self)`
  - `mostraTraiettoria3D()`   

  - **Galton3Dcorr** `(Galton3D + matriceCorrelazione)`  
    - `simula3D(self, stampa)`  
    - `mostra3D(self)`
    - `proiezioni3D(self, stampa)`  (ereditata da Galton3D)
    - `mostraMatrice(self)`         (ereditata da Galton3D)
    - `mostraTraiettoria3D()`       (ereditata da Galton3D)

<br>

Il simbolo "_" indica un **parametro esterno** della classe, ovvero un valore che non viene direttamente memorizzato all'interno di ***self***.

## Prerequisiti

In questa sezione verranno descritte i prerequisiti per clonare ed eseguire il codice Python. Tra parentesi saranno indicate le versioni utilizzate durante la scrittura del codice.

Avere istallato sul dispositivo:
- ***Python3*** (3.12.7) con:
    - *pip3* (24.3.1)
        - NumPy (2.0.2)
        - SciPy (1.14.1)
        - Matplotlib (3.9.2)
        - tqdm (4.67.1)
        - ArgParse (1.4.0)
- ***Git*** (2.39.5 Apple Git-154)

OPZIONALE: LaTeX (pdfTeX 3.141592653-2.6-1.40.24 - TeX Live 2022) per la visualizzazione di testi LaTeX sui grafici.

***pip3*** è un gestore di pacchetti utile ad installare, aggiornare e rimuovere pacchetti di Python3.

Il modulo ***ArgParse*** è parte della libreria standard a partire dalla versione 3.2 di Python.

Per la verifica della corretta installazione di ***Python***, ***pip3*** e ***Git***, si possono eseguire le seguenti righe di comando da Terminale:

```bash
python3 --version
pip3 --version
git --version
```

NB: i comandi possono differire in base al sistema operativo o alla configurazione in uso. Ciononostante la struttura di base del comando è la stessa.

### Python

Se Python3 non è installato, il lettore curioso viene invitato a consultare il sito ufficiale di [Python](https://www.python.org). Si potrà installare Python tramite eseguibile oppure tramite un sistema di gestione dei pacchetti (ad esempio Homebrew per Mac, Winget per Windows e APT per Linux).

### Librerie di Python

NB: per procedere, Python deve essere disponibile al sistema.

Per verificare la corretta installazione delle librerie di Python, si deve fare download del file "*verificaLibrerie.py*" e lanciare il seguente codice:

```bash
python3 verificaLibrerie.py
```

Se verrà visualizzato la riga "Tutte le librerie sono installate!", allora le librerie sono disponibili e pronte all'uso.

Invece, se delle librerie non sono installate, si deve fare il download del file "*requirements.txt*" ed eseguire il seguente comando: 

```bash
pip3 install -r requirements.txt
```

NB: A partire da Python 3.4, ***pip3*** è incluso come parte del pacchetto standard di Python.

### Git

L'installazione di Git può essere effettuata seguendo i passaggi descritti sul sito ufficiale di [git](https://git-scm.com).


### Clonazione della repository

Il codice è completamente disponibile nella repository GitHub "ProgettoMCF" dell'autore "filippo-tintori".

Dopo aver installato le dipendenze necessarie, il progetto (cartella con all'interno file) può essere clonato in locale col seguente codice:

```bash
git clone https://github.com/filippo-tintori/ProgettoMCF.git
```

## Note preliminari all'esecuzione

I **print** nel codice fanno uso della codifica di caratteri Unicode ***UTF-8***.
Quindi si prega di controllare che il sistema sia adeguato a stampare questo tipo di caratteri, altrimenti non verranno visualizzati correttamente.

I grafici sono stati studiati per far si che il rendering del testo dei grafici fosse gestito dal motore LaTeX, così da avere una scrittura più appagante esteticamente e più professionale. 

Se LaTeX non è disponibile, si può comunque eseguire il codice, commentando il codice seguente presente nei file "*galton.py*" e "*studio2D.py*":

```bash
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
```

## Esecuzione del programma

L'esecuzione del codice è gestita dal modulo ArgParse con il quale l'utente può decidere che tipo di Macchina di Galton simulare (2D, 3D o 3D correlata) con le relative caratteristiche instrinseche.

Per conoscere gli argomenti di ArgParse da utilizzare per il codice, si consiglia di eseguire il comando di **help**:

```bash
python3 main.py --help
# equivalente a:
python3 main.py -h
```

Esempi di utilizzo:

```bash
# 2 dimensioni
python3 main.py --dim2 --nPassi 20 --probX 0.4 --margineErrore 0.01

# 3 dimensioni
python3 main.py --dim3 --nPassi 20 --probX 0.4 --probY 0.4 --margineErrore 0.01

# 3 dimensioni correlate
python3 main.py --dim3corr --nPassi 20 --probX 0.4 --probY 0.5 --margineErrore 0.01 --matrice 1 0.9 0.9 1 

# studio 2D
python3 main.py --studio2D
# studio 3D
python3 main.py --studio3D
# studio 3D corr
python3 main.py --studio3Dcorr
```

Per maggiori informazioni sugli argomenti di ArgParse, si può usare il comando di help:
```bash
python3 main.py --help
```


## Simulazioni

Con il codice sviluppato è possibile eseguire simulazioni con qualsiasi combinazione di input, ovviamente sempre nel limite del ragionevole.

Ogni simulazione ha bisogno di argomenti caratteristici per la simulazione.

Simulazione 2D:
- ***dim2***
- ***nPassi*** (int)
- ***margineErrore*** (float) o ***nPalle*** (int)
- ***probX*** (float)

Simulazione 3D:
- ***dim3***
- ***nPassi*** (int)
- ***margineErrore*** (float) o ***nPalle*** (int)
- ***probX*** (float)
- ***probY*** (float)

Simulazione 3D con correlazione:
- ***dim3corr***
- ***nPassi*** (int)
- ***margineErrore*** (float) (float)o ***nPalle*** (int)
- ***probX*** (float)
- ***probY*** (float)
- ***matrice*** (matrice di correlazione 2x2)


È consigliato utilizzate il parametri ***margineErrore*** così da far calcolare al programma direttamente un numero adeguato di palline da simulare e da garantire che la distribuzione delle palline segua un'approssimazione gaussiana della distribuzione binomiale, riducendo l'incertezza nella posizione media.

## Studio delle simulazioni 

Le simulazioni prese in cosiderazione per lo studio sono state scelte per avere una panoramica dei risultati che si ottengono e del loro cambiamento in base ai parametri in input.

Per far vedere lo studio effettuato sono stati disposti degli argomenti di ArgParse:


### Studio 2D
Avviabile con:
```bash
python3 main.py --studio2D
```

Questo studio è stato diviso in 3 parti.

La **prima parte** implementa un metodo Monte Carlo per studiare la distribuzione delle media $\mu$ ottenuta dalla macchina di Galton. La ripetizione della simulazione serve a dimostrare la robustezza dei risultati, convalidare il modello teorico, evidenziare la convergenza statistica e verificare che il numero di palline scelto è adeguato. Aumentando il numero di ripetizioni, gli effetti delle fluttuazioni casuali diminuiscono e i risultati diventano più stabili.

La **seconda parte** esegue un'analisi statistica di un processo di Galton considerando sia la distribuzione binomiale che gaussiana, e osservando come queste distribuzioni evolvono *al variare del numero di passi*. Per comprendere la transizione da una distribuzione binomiale a una distribuzione gaussiana, principio fondamentale  della teoria del limite centrale.

La **terza parte** permette di analizzare l'evoluzione della distribuzione delle posizioni finali *al variare della probabilità di andare a DX* in un sistema con 500 passi. 

### Studio 3D
Avviabile con:
```bash
python3 main.py --studio3D
```

Questo studio è diviso in 3 parti.

La **prima parte** esegue un'analisi statistica della macchina di Galton 3D per comprendere come il sistema evolva in funzione del numero di passi. Aumentando il numero di passi l'approssimazione della gaussiana alla binomiale è sempre più evidente. Per un piccolo numero di passi, la distribuzione binomiale è la migliore approssimazione. Invece, per un numero grande di passi, la distribuzione gaussiana diventa una buona approssimazione (conferma del Teorema del Limite Centrale).

La **seconda parte** analizza l'evoluzione dei risultati variando la probabilità di andare a destra lungo l'asse X, mantenendo invariata la probabilità di andare a destra lungo l'asse Y.

La **terza parte** analizza l'evoluzione dei risultati variando la probabilità di andare a destra lungo l'asse Y, mantenendo invariata la probabilità di andare a destra lungo l'asse X.


### Studio 3D correlato
Avviabile con:
```bash
python3 main.py --studio3Dcorr
```

Si analizza qualitativamente come diverse matrici di correlazioni influenzano la distribuzione delle posizioni finali.


## Spiegazione delle decisioni prese

In questa sezione di daranno alcune spiegazioni delle decisioni prese.

### Spiegazione generale del codice

Le classi con parametri e metodi sono state utilizzate per rendere il codice più modulare ed efficiente, consentendo di incapsulare comportamenti fisici in oggetti. Questo approccio migliora la manutenibilità e la riusabilità del codice, oltre a garantire una gestione sicura delle variabili, permettendo ai metodi di operare su di esse senza interferenze esterne. Gli array (`numpy`) sono preferiti alle liste di Python per la loro maggiore efficienza nelle operazioni numeriche: gli array consentono operazioni vettorializzate, che sono molto più veloci rispetto alle iterazioni sulle liste, e occupano meno memoria, ottimizzando le prestazioni quando si lavora con grandi quantità di dati. Inoltre, l'uso di `curve_fit` consente di stimare i parametri di una funzione in modo efficiente, applicando il principio dei minimi quadrati per ridurre l'errore tra modello e dati, ed è particolarmente utile per relazioni non lineari o complesse, migliorando così l'analisi dei dati. 

Per generare numeri casuali uniformemente distribuiti tra 0 e 1, si utilizza `np.random.rand`, fondamentale per simulazioni stocastiche come la distribuzione delle palline nella macchina di Galton. Grazie alla velocità di `numpy` nella generazione di numeri casuali, il processo diventa estremamente efficiente, specialmente quando si devono simulare grandi quantità di palline. Inoltre, il metodo `norm.pdf` di un oggetto di distribuzione casuale calcola la funzione di densità di probabilità in un dato punto, utile per determinare la probabilità di osservare un valore specifico secondo la distribuzione associata, permettendo un'analisi più precisa dei dati. La distribuzione `multivariate_normal` è stata scelta per descrivere variabili casuali multidimensionali correlate, come nel caso delle palline della macchina di Galton 3D correlata, che interagiscono tra di loro in un modo che può essere rappresentato da una distribuzione normale multidimensionale, rendendo così la simulazione più realistica e adeguata a comportamenti complessi. Infine, il numero di palline utilizzato nelle simulazioni influisce sulla precisione della distribuzione finale: un numero maggiore di palline riduce l'errore statistico e offre una rappresentazione più accurata del comportamento atteso.

### Numero di palline adeguato per la simulazione

Per determinare un numero adeguato di palline da utilizzare nella simulazione di una macchina di Galton, sono stati effettuati calcoli preliminari basati sull'**approssimazione gaussiana della distribuzione binomiale**. Di seguito, vengono riportati i dettagli del procedimento.

Considerando una distribuzione binomiale con probabilità $p$ e numero di livelli $n_{\text{passi}}$\, la media $\mu$ e la deviazione standard $\sigma$ della distribuzione delle palline sono date da:

$$
\mu = n_{\text{passi}} \cdot p
$$
$$
\sigma = \sqrt{n_{\text{passi}} \cdot p \cdot (1 - p)}
$$

Per garantire che l'incertezza nella stima della posizione media sia sufficientemente piccola, imponiamo che la deviazione standard della media sia inferiore o uguale a una soglia prefissata ($\delta$). La deviazione standard della media è espressa come:


$$
\frac{\sigma}{\sqrt{n_{\text{passi}}}} \leqslant \delta
$$
da cui segue la condizione:
$$
n_{\text{passi}} \geqslant \frac{\sigma^2}{\delta^2}
$$
assicurando così un fit binomale e gaussiano adeguato. In quanto si minimizza l'inccertezza nella posizione media delle palline, e quindi una minimizzazione dell'errore relativo nella stima della distribuzione.

Nel caso di una macchina di Galton 2D, si considera la distribuzione delle palline lungo l'unico asse. 

Nel caso di una macchina di Galton 3D, in cui si hanno due assi (X e Y), per determinare il numero totale di palline necessarie, si calcola separatamente il numero di palline richiesto per ciascun asse $n_{\text{palline}, X}$ e $n_{\text{palline}, Y}$ utilizzando la formula sopra riportata. Successivamente, si seleziona il valore maggiore tra i due:

$$
n_{\text{palline}} = \max(n_{\text{palline}, X}, \ n_{\text{palline}, Y})
$$

Questo criterio garantisce che la simulazione sia accurata in tutte le dimensioni considerate.

### Uso di scipy.special.comb

La funzione ***scipy.special.comb*** è stata usata per calcolare il coefficiente binomiale previsto nella omonima distribuzione. Quando usata in modalità esatta, questa poteva causare errori di overflow alla funzione, per cui è stato inserito un "*except*" che dirotta la funzione verso il calcolo non esatto in caso di warning. Il parametro "exact=False" è capace di rimuovere l'overflow tramite approssimazioni matematiche.

### Simulazione dei passi dipendenti dalla correlazione

Si applica la decomposizione di Cholesky per ottenere una matrice triangolare inferiore. Questa matrice permette di trasformare un set di passi generati casualmente, introducendo la correlazione desiderata. Così facendo, i passi diventano dipendenti dalla matrice immessa. Successivamente, i passi vengono confrontati con una soglia di probabilità per determinare se un dato passo si traduce in uno spostamento effettivo verso destra.

Dal punto di vista matematico, la correlazione viene introdotta tramite un prodotto lineare (*np.einsum*) tra la matrice di Cholesky e ogni insieme di passi indipendenti per ogni pallina.

### Calcolo della matrice di correlazione stimata

Per ricalcolare la matrice di correlazione dai passi verso destra effettuati lungo l'asse X e lungo l'asse Y, si usa la funzione *np.corrcoef*, che calcola la matrice di correlazione tra i vettori di spostamento finale lungo X e Y.


### Calcolo del $\chi^2$

È una misura statistica utilizzata per confrontare i dati osservati con quelli attesi, al fine di verificaare se esiste una differenza significatica tra essi. Viene calcolato come:

$$
\chi^2=\sum_{k=1}^{n} \frac{(O_k - E_k)^2}{E_k}
$$
con:
$$
O_k = \text{numero di volte che il valore $k$ è stato osservato / rilevato}

\newline

E_k = \text{numero di volte che il valore $k$ è atteso}
$$

Causa problemi nel caso in cui il valore atteso $E_k$ è pari a 0. 

Inizialmente si è provato ad allargare i bins fino a quando tutti questi avessero tutti numeri attesi maggiori di 0, ma questo si è visto capace di ridurre drasticamente l'efficienza del codice e non utile ai fini del progetto.

Per cui si è scelto di non includere i punti problematici, anche se da un punto di vista statistico sarebbe stato meglio proseguire in diverso modo.

Nonostante ciò, dagli studi sulle diverse macchine di Galton si può vedere come questo metodo risponda correttamente al nostro scopo.



## Referenze

1. Introduzione all'analisi degli errori di John R. Taylor
2. [it.wikipedia.org](https://it.wikipedia.org/wiki/Macchina_di_Galton)
3. [cirdis.stat.unipg.it](https://web.archive.org/web/20121117140646/http://cirdis.stat.unipg.it/files/macchina_galton/macchina_galton/index.html)
4. [https://www.dm.unibo.it](https://www.dm.unibo.it/pls/mma-cdl-edition/Senza%20Nome/progetto-GALTON3D/galton3d.html)


## Licenza

Il progetto è sotto licenza MIT. Per maggiori informazioni, si consiglia di visualizzare il file **LICENSE** presente nella reposotory.