# Seeds Dataset – KMeans ML Pipeline

Questo progetto, basato sul [dataset Seeds](https://archive.ics.uci.edu/dataset/236/seeds), 
implementa una pipeline completa di Machine Learning non supervisionato, 
confrontando due approcci di riduzione delle feature:

- PCA (Principal Component Analysis)
- Multicollinearità (rimozione delle feature correlate)

## Obiettivo

Analizzare e clusterizzare il dataset Seeds utilizzando K-Means, 
confrontando due strategie di preprocessing per capire quale produce cluster più separati e coerenti.

## Avvio dell'applicazione

### Requisiti

Prima di avviare l'applicazione, assicurati di avere installato tutte le dipendenze necessarie. Puoi trovarle nel file `requirements.txt`.

### Avvio in locale

Per visualizzare l'API in locale è necessario recarsi al seguente indirizzo:

   ```
   http://127.0.0.1:5000/
   ```
## Avvio con Docker

### Prerequisiti

Assicurati di avere installato Docker. Puoi scaricare l'applicazione [qui](https://www.docker.com/products/docker-desktop/).

Verifica l’installazione con:

```bash
docker --version
```

### Utilizzo di Docker

1. Costruisci l'immagine Docker da linea di comando:
   ```bash
   docker build -t seeds-kmeans .
   ```
   
2. Si possono controllare le informazioni dell'immagine appena creata con il comando:
   ```bash
   docker image ls
   ```

3. Costruisci e avvia il container da linea di comando (copiando questo comando, verrà chiamato "seeds"):
   ```bash
   docker run --name seeds -p 5000:5000 seeds-kmeans
   ```
   Se l'operazione è andata a buon fine è possibile vedere lo stavo attivo del container tramite il comando:
   ```bash
   docker ps
   ```

4. Accedi all'applicazione (nella sua route home) tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```

## Output generati

Il progetto salva automaticamente, a seconda degli endpoint utilizzati, i seguenti grafici:

- Correlation matrix (prima/dopo riduzione)
- Distribuzioni feature 
- Boxplot outlier 
- PCA explained variance 
- PCA 2D projection 
- Elbow curve K-Means 
- Visualizzazione cluster 2D 
- Report confronto modelli

Tutti i grafici vengono salvati in:
   ```
   /figures
   ```
### Output main con Pipeline Overview

Se lanciato dal terminale, la pipeline di progetto esegue automaticamente:

1. Caricamento dataset
2. Exploratory Data Analysis (EDA)
3. Rilevamento outlier (metodo IQR)
4. Pulizia dati
5. Robust Scaling delle feature 
6. Riduzione dimensionale con PCA 
7. Rimozione multicollinearità
8. Clustering con K-Means
9. Valutazione modelli e confronto finale

## Utilizzo dell'API

L'API è consultabile direttamente tramite dei plug-in browser. Due esempi di essi sono i seguenti:

- Rest-Client (Chrome): [download](https://chromewebstore.google.com/detail/rest-client/oienkoejnhkbcibhdnpjoemdnmiokgah)
- Rested (Firefox): [download](https://addons.mozilla.org/en-US/firefox/addon/rested/)

## Funzionamento degli Endpoints

Il progetto è provvisto di diversi endpoint, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```

Nella home sono spiegati i due passaggi da seguire:

**1. Caricamento del dataset e ritiro dell'uuid**

**2. Utilizzare gli altri endpoint inserendo l'uuid in fondo**

## Inserimento del dataset da consultare

La prima cosa da fare sull'api è recarsi all'endpoint POST
```
http://127.0.0.1:5000/load
```

e caricare un dataset, che nel nostro caso è "Seeds":
```json
{"file_path": "data/seeds_dataset.txt"}
```

all'interno della risposta (riga 14) sarà contenuto un "pipeline-id", che è un uuid
che identifica univocamente l'utente che lavora al codice: è necessario per tracciare le modifiche
che fa un utente piuttosto che un altro

**L'utilizzo dell'UUID è fondamentale in quanto senza di esso gli altri endpoint non funzionerebbero**

Una volta inserito il dataset e copiato l'uuid si può procedere a consultare gli altri endpoints

## Endpoints per consultare il dataset

Tutti questi endpoints dovranno essere seguiti da una UUID. Un esempio è "8de3c40a-b610-4866-836c-d12cc2667dce"

### Summary [GET]

Genera un riepilogo statistico del dataset. 
L'endpoint esegue un'analisi esplorativa (EDA) sui dati della pipeline, producendo:

- Statistiche descrittive delle feature numeriche (count, mean, std, min, max, percentili, ecc.).
- Un riepilogo automatico dell'analisi esplorativa generato dall'EDAAnalyzer.
```
http://127.0.0.1:5000/summary
```

### Correlation [POST]

L'endpoint calcola la correlazione tra le variabili numeriche e crea una visualizzazione grafica sotto forma di heatmap/matrice di correlazione.

L'immagine generata viene salvata nella directory outputs
```
http://127.0.0.1:5000/correlation
```

### Distributions [POST]

L'endpoint crea una serie di grafici che mostrano la distribuzione 
statistica delle feature presenti nel dataset

L'immagine generata viene salvata nella directory outputs
```
http://127.0.0.1:5000/distributions
```

### Boxplots [POST]

Genera i boxplot delle feature per l'identificazione degli outlier.
L'endpoint produce una visualizzazione tramite boxplot per ciascuna feature.

L'immagine generata viene salvata nella directory outputs
```
http://127.0.0.1:5000/boxplots
```
### Outliers Remove [POST]

Rimuove gli outlier dal dataset utilizzando il metodo IQR (Interquartile Range).
```
http://127.0.0.1:5000/outliers/remove
```

### Outliers Distributions [POST]

Genera un confronto grafico tra distribuzioni prima e dopo la rimozione degli outlier.

Questo endpoint visualizza l’effetto della rimozione degli outlier confrontando:

Distribuzione del dataset originale
Distribuzione del dataset pulito (IQR filtering)

È utile per verificare visivamente l’impatto della pulizia dei dati.
```
http://127.0.0.1:5000/outliers/distributions
```