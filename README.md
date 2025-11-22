# Machine learning Final Project
Studio del dataset student-por per analizzare i fattori che influenzano il voto finale (G3) e costruire modelli di regressione (Random Forest, SVR) ottimizzati con RandomizedSearchCV.

## 📘 Struttura dettagliata del progetto

### 1. Setup dell’ambiente e import delle librerie
Configuro l’ambiente di lavoro importando le librerie necessarie per analisi e modellazione:
- `pandas`, `numpy` per manipolazione dati  
- `matplotlib`, `seaborn` per visualizzazioni  
- `scikit-learn` per preprocessing, modelli e validazione  

Imposto anche opzioni di stile e un seed per garantire riproducibilità.

---

### 2. Caricamento e ispezione iniziale del dataset
Carico il file `student-por.csv` e verifico:
- prime righe del dataset  
- struttura (shape, tipi di variabili, categorie)  
- presenza di valori mancanti  
- coerenza generale dei dati  

Questa fase introduce la struttura del dataset e permette di individuare eventuali criticità iniziali.

---

### 3. Analisi Esplorativa dei Dati (EDA)

#### 3.1 Distribuzioni e variabili principali
Analizzo la distribuzione dei voti (`G1`, `G2`, `G3`) tramite istogrammi e boxplot per identificare:
- presenza di outlier  
- andamento generale delle prestazioni degli studenti
  <img width="407" height="311" alt="image" src="https://github.com/user-attachments/assets/ffd7d382-6ac6-43f9-87f9-5868a2b93700" />


#### 3.2 Matrice di correlazione
Creo una heatmap delle correlazioni per evidenziare relazioni rilevanti tra variabili numeriche.  
Osservo in particolare la relazione forte tra `G1`, `G2` e `G3`.

<img width="657" height="497" alt="image" src="https://github.com/user-attachments/assets/24ef26d8-fb5f-4b32-ae8f-1c3fa3ed7d97" />


#### 3.3 Analisi di gruppi
Studio come cambia `G3` in base a:
- consumo settimanale di alcol (`Dalc`)  
- consumo nel weekend (`Walc`)  
- supporto scolastico (`schoolsup`)  
- assenze (`absences`)  
- tempo libero, salute e relazioni sociali  

Questa parte individua pattern utili per capire quali variabili possono essere predittive.

---

### 4. Preparazione dei dati (Preprocessing)

#### 4.1 Codifica variabili categoriche
Trasformo le variabili categoriche in variabili numeriche tramite One-Hot Encoding, per rendere i dati compatibili con i modelli.

#### 4.2 Scaling delle feature
Applico lo scaling (StandardScaler) nei modelli sensibili alla scala (es. SVR).  
I Random Forest non obbligano lo scaling, ma mantengo uniformità tra gli esperimenti.

#### 4.3 Suddivisione del dataset
Divido i dati in:
- **Training set (80%)**
- **Test set (20%)**

per valutare i modelli in maniera imparziale ed evitare leakage.

---

### 5. Impostazione dei due scenari di modellazione

#### Scenario A — Senza G1 e G2
Prevedo `G3` utilizzando solo variabili socio-familiari, personali e scolastiche **senza i voti intermedi**.  
Simula un contesto realistico dove i voti precedenti non sono disponibili.

#### Scenario B — Con G1 e G2
Aggiungo `G1` e `G2` alle feature.  
Serve per valutare quanto i voti intermedi migliorano la previsione del voto finale.

---

### 6. Modelli di Machine Learning

#### 6.1 Baseline
Creo una baseline che predice sempre la media del training set.  
Serve come riferimento minimo: ogni modello deve superarla per essere considerato utile.

---

#### 6.2 Random Forest Regressor
Modello non lineare basato su decision tree aggregati.  
Iperparametri ottimizzati:
- `n_estimators`  
- `max_depth`  
- `min_samples_split`  
- `min_samples_leaf`  
- `max_features`  

È robusto, gestisce interazioni tra variabili e permette di stimare l’importanza delle feature.

---

#### 6.3 Support Vector Regression (SVR)
Modello basato su margini ottimizzati e kernel non lineari.  
Iperparametri ottimizzati:
- `C`  
- `epsilon`  
- `kernel` (lineare/RBF)  
- `gamma`  

Risulta potente ma molto sensibile allo scaling e alla scelta dei parametri.

---

### 7. Ottimizzazione tramite RandomizedSearchCV
Utilizzo `RandomizedSearchCV` per ricercare iperparametri ottimali.

#### Perché Random Search?
- esplora parametri in modo casuale → più efficiente della grid search  
- permette di testare uno spazio di ricerca molto più ampio  
- con meno tempo trova spesso soluzioni migliori  
- evita di esplorare solo combinazioni rigidamente predefinite  

Ogni modello viene validato tramite cross-validation per ridurre il rischio di overfitting.

---

### 8. Valutazione dei modelli

#### Metriche utilizzate
- **MAE** (errore medio in punti di voto)  
- **RMSE** (errore pesato, penalizza errori grandi)  
- **R²** (variabilità spiegata dal modello)  

#### Analisi degli errori
- confronto tra predizioni e valori reali  
- analisi dei residui  
- identificazione degli studenti su cui il modello sbaglia di più  

Serve per capire pattern nascosti e limiti dei modelli.

---

### 9. Conclusioni e spunti finali
Riassumo i risultati principali:
- quali modelli hanno performato meglio  
- differenze tra Scenario A e B  
- ruolo cruciale dei voti intermedi  
- influenza delle variabili sociali e familiari  
- possibili applicazioni per interventi educativi mirati  

Indico anche possibili sviluppi futuri:
- utilizzo di modelli più avanzati (XGBoost, LightGBM)  
- interpretabilità tramite SHAP  
- analisi per sottogruppi (studenti a rischio, frequenza assenze, ecc.)

