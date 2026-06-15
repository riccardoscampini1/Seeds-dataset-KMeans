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

L'API è consultabile direttamente da browser.

Se dovesse servire, è possibile installare dei plug-in, tra cui:

- Rest-Client (Chrome): [download](https://chromewebstore.google.com/detail/rest-client/oienkoejnhkbcibhdnpjoemdnmiokgah)
- Rested (Firefox): [download](https://addons.mozilla.org/en-US/firefox/addon/rested/)

## API Endpoint

Il progetto è provvisto di diversi endpoint, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```
Le funzionalità dell'API sono le seguenti:
