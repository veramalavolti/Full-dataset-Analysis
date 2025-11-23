# Machine learning Final Project
Studio del dataset student-por per analizzare i fattori che influenzano il voto finale (G3) e costruire modelli di regressione (Random Forest, SVR) ottimizzati con RandomizedSearchCV.

Questo progetto applica metodi di data science al **Student Performance Dataset (UCI)** per analizzare i fattori che influenzano il rendimento scolastico e costruire modelli predittivi accurati.

L’analisi comprende:
- Exploratory Data Analysis (EDA)
- Preprocessing con Pipeline e ColumnTransformer
- Confronto di modelli supervisionati (Linear Regression, Random Forest, SVR)
- Tuning degli iperparametri con RandomizedSearchCV
- Interpretazione dei risultati tramite feature importance

## Il dataset e gli obiettivi
<img width="1002" height="564" alt="image" src="https://github.com/user-attachments/assets/1c8d3897-9b13-40a4-8f49-d1f4366b8330" />

- **Fonte:** UCI Machine Learning Repository  
- **File usato:** `student-por.csv` (corso di lingua portoghese)  
- **Dimensioni:** 649 studenti × 33 variabili  

### Tipologie di variabili
- **Famiglia/Sociale:** istruzione dei genitori, occupazione, dimensione familiare  
- **Personale/Comportamentale:** età, genere, attività sociali, consumo di alcol  
- **Scolastiche:** tempo di studio, assenze, supporto educativo  
- **Performance:** G1, G2, G3 (voti ai diversi periodi)

Questo dataset integra aspetti sociali, comportamentali e scolastici, rendendolo ideale per analisi descrittive e predittive.

<img width="1002" height="564" alt="image" src="https://github.com/user-attachments/assets/aaa9eb3b-620e-40c9-80d5-9ffd1e941fc3" />

### Obiettivi del progetto
- Predire il **voto finale (G3)** tramite modelli di regressione.
- Confrontare le performance di diversi algoritmi.
- Ottimizzare gli iperparametri mediante **RandomizedSearchCV + Nested Cross-Validation**.
- Individuare i fattori che influenzano maggiormente il rendimento.
- Derivare insight utili per supportare studenti in contesti svantaggiati.

L’obiettivo è unire rigore tecnico e utilità pedagogica.


## Struttura dettagliata del progetto

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

<img width="1002" height="564" alt="image" src="https://github.com/user-attachments/assets/de712f76-0ddb-4432-9beb-75fa1794efd6" />

### Risultati principali dell’EDA
- **Correlazioni forti:**  
  - G3 è strettamente correlato con **G1 (0.80)** e **G2 (0.91)**.
- **Fattori positivi:** tempo di studio, istruzione dei genitori.  
- **Fattori negativi:** fallimenti passati, consumo di alcol.  
- **Distribuzione di G3:** centrata attorno all’11–12, ma con ampia variabilità (0–19).


---

# 🟣 **Slide 5 — Metodologia**
<img width="1405" height="786" alt="image" src="https://github.com/user-attachments/assets/be9a11bd-0d59-4b43-a15d-6161a4a73090" />

### 🧪 Workflow

1. **EDA**
   - Analisi delle distribuzioni, pattern e correlazioni.

2. **Preprocessing**
   - Numeriche → imputazione + standard scaling  
   - Categoriali → imputazione + One-Hot Encoding  
   - Implementato con `ColumnTransformer` + `Pipeline`.

3. **Scenari**
   - **A:** include G1 e G2 (scenario benchmark)  
   - **B:** esclude G1 e G2 (scenario realistico di inizio anno)

4. **Modelli**
   - Linear Regression  
   - Random Forest Regressor  
   - Support Vector Regressor

5. **Hyperparameter Tuning**
   - RandomizedSearchCV + Nested Cross-Validation

6. **Valutazione**
   - MAE, RMSE, R²

7. **Interpretazione**
   - Permutation Importance per capire le variabili più rilevanti

pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor())
])



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
<img width="1440" height="807" alt="image" src="https://github.com/user-attachments/assets/604597d5-21e0-45cc-ab18-acb5abe56d0c" />

---

### 9. Conclusioni e spunti finali
<img width="1440" height="807" alt="image" src="https://github.com/user-attachments/assets/dba388fb-f3fc-440e-a0b8-344639e61b6e" />

Riassumo i risultati principali:
- quali modelli hanno performato meglio
- differenze tra Scenario A e B  
- ruolo cruciale dei voti intermedi  
- influenza delle variabili sociali e familiari
<img width="1440" height="807" alt="image" src="https://github.com/user-attachments/assets/e2a6b49d-de03-4aeb-b92d-108586daf03e" />

- possibili applicazioni per interventi educativi mirati



Indico anche possibili sviluppi futuri:
- utilizzo di modelli più avanzati (XGBoost, LightGBM)  
- interpretabilità tramite SHAP  
- analisi per sottogruppi (studenti a rischio, frequenza assenze, ecc.)
<img width="1440" height="807" alt="image" src="https://github.com/user-attachments/assets/43bedcce-6853-4f7e-803c-1049f8a48b0b" />


