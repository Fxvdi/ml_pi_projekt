# Methodeneignung: Eingrenzung der Anomalieerkennungs-Methoden und AutoML-Frameworks für den TEP-Datensatz

**Zweck dieses Dokuments:** Noch *kein* Modellvergleich, sondern eine
systematische Eingrenzung: Welche Methodenfamilien und AutoML-Frameworks kommen
für die vorliegenden Daten überhaupt infrage – und welche scheiden aus? Die
Begründung erfolgt strikt aus den Dateneigenschaften heraus und wird, wo
möglich, mit wissenschaftlichen Quellen belegt. Der empirische Vergleich der
verbleibenden Kandidaten folgt in den Notebooks (`01_…`, `02_…`).

---

## 1. Empirische Dateneigenschaften

Alle Punkte sind direkt aus den Daten verifiziert (siehe `eda_dataset.ipynb`
und `tep/data.py`); der Datensatz ist die erweiterte TEP-Simulation von Rieth
et al. [2], basierend auf dem Prozessmodell von Downs & Vogel [1].

| # | Eigenschaft | Befund in den Daten |
|---|---|---|
| D1 | **Tabular-numerisch, vollständig** | 52 kontinuierliche Größen (41 Messgrößen `xmeas`, 11 Stellgrößen `xmv`), keine fehlenden Werte, keine kategorialen Features |
| D2 | **Stark kreuzkorreliert** | Ausgeprägte Korrelationsblöcke zwischen den Sensoren (Korrelations-Heatmap in der EDA); physikalisch gekoppelte Prozessgrößen |
| D3 | **Zeitreihe, äquidistant** | Abtastung alle 3 min; Fehler wirken dynamisch (u. a. `IDV(13)` *Slow Drift*, `IDV(14/15)` *Sticking*) |
| D4 | **Viele kurze, unabhängige Läufe** | 500 Läufe × 500 Samples (Training) bzw. × 960 Samples (Test) je Zustand – *nicht* eine lange durchgehende Serie |
| D5 | **Saubere Normaldaten separat verfügbar** | `TEP_FaultFree_Training`: 250 000 Samples ausschließlich Normalbetrieb (faultNumber = 0) |
| D6 | **Labels vorhanden (für Evaluation)** | Fehlertyp je Lauf + bekannter Fehlereintritt (Training: Sample 20, Test: Sample 160) → Per-Sample-Labels ableitbar |
| D7 | **Großes Volumen** | ~15,3 Mio. Zeilen gesamt; quadratisch skalierende Verfahren brauchen Subsampling |
| D8 | **Ein Betriebspunkt, synthetisch, stationär** | Normalbetrieb ist ein einzelner stationärer Arbeitspunkt (Base Case aus [1]); kein Multimode-Betrieb, kein Konzeptdrift |
| D9 | **Anomalien sind persistent** | Fehler sind dauerhafte Zustandswechsel ab Fehlereintritt, keine punktuellen Ausreißer |

---

## 2. Was folgt daraus? (Anforderungen an die Methoden)

Aus den Dateneigenschaften ergeben sich direkt die Auswahlkriterien:

- **D1 → keine komplexen Feature-Pipelines nötig.** Standardisierung genügt;
  AutoML-Stärken wie Kategorien-Encoding oder Missing-Value-Handling bringen
  hier keinen Mehrwert.
- **D2 → multivariate Methoden sind Pflicht.** Univariate Regelkarten
  ignorieren die Korrelationsstruktur, in der sich viele Fehler überhaupt erst
  zeigen; das ist das Kernargument der multivariaten statistischen
  Prozessüberwachung (MSPC) [3, 4]. (Empirisch in Phase 1 bestätigt:
  Mahalanobis > univariates max|z|.)
- **D3 → zeitbewusste Methoden sind plausibel, aber nicht zwingend.** Viele
  TEP-Fehler äußern sich als Niveauverschiebung und sind punktweise erkennbar;
  Drift- und Sticking-Fehler sowie die notorisch schweren Fälle (IDV 3, 9, 15
  [4, 6]) sprechen zusätzlich für dynamische Erweiterungen (DPCA [5], CVA [4]).
- **D4 → Methoden für *eine* lange Serie passen nicht.** Verfahren wie
  Matrix-Profile/Discord-Discovery [15] setzen eine durchgehende Zeitreihe
  voraus; die Run-Struktur (500 unabhängige Wiederholungen) passt besser zu
  punkt- bzw. fensterbasierten Detektoren mit Run-weiser Auswertung.
- **D5 → Semi-supervised Novelty Detection ist das natürliche Paradigma.**
  Da fehlerfreie Trainingsdaten garantiert sauber sind, lernen One-Class-
  Verfahren die Normalverteilung ohne Kontaminationsannahme [7, 16]. Das ist
  die in der Taxonomie von Chandola et al. [16] beschriebene
  „semi-supervised anomaly detection".
- **D6 → das zentrale AutoML-Problem der Anomalieerkennung ist hier lösbar.**
  Unsupervised Modellselektion ohne Labels ist ungelöst bzw. unzuverlässig
  [13, 14]. Weil TEP Labels *zur Validierung* liefert, kann Modell- und
  Hyperparameterselektion supervised erfolgen – ein strukturierter Benchmark
  mit automatischer Selektion ist damit methodisch sauber möglich.
- **D7 → Skalierbarkeit als hartes Kriterium.** Kernel-OCSVM (O(n²)–O(n³))
  und LOF sind nur mit Subsampling praktikabel; lineare Verfahren (PCA, ECOD
  [9], IsolationForest [8]) skalieren problemlos.
- **D8 → keine Multimode-/Adaptionsverfahren nötig.** Methoden für
  Betriebspunktwechsel oder Online-Adaption adressieren ein Problem, das in
  diesen Daten nicht existiert.
- **D9 → Bewertung muss Persistenz berücksichtigen.** Detection Delay und
  False Alarm Rate sind aussagekräftiger als reine Punkt-Metriken; Wu & Keogh
  [17] warnen explizit vor naiven Punkt-Benchmarks in der
  Zeitreihen-Anomalieerkennung.

---

## 3. Eingrenzung der Methodenfamilien

### 3.1 Geeignet

| Familie | Vertreter | Begründung (Daten-Bezug) |
|---|---|---|
| **MSPC / Latent-Variable** | PCA mit T²/SPE [3], Dynamic PCA [5], CVA [4] | Direkt für D2 (Korrelation) und D8 (ein stationärer Arbeitspunkt) konstruiert; etablierter TEP-Standard und Literatur-Referenz [4, 6] |
| **One-Class / Distanz / Dichte** | OCSVM [7], kNN, LOF [10], IsolationForest [8], ECOD [9] | Passen exakt zum semi-supervised Setting (D5); verteilungsfrei, nichtlinear; in breiten Vergleichsstudien wettbewerbsfähig [11, 12] |
| **Rekonstruktionsbasiert** | Autoencoder, VAE [18, 19] | Lernen die Normal-Mannigfaltigkeit nichtlinear (D2, D5); Rekonstruktionsfehler als Score; Standardansatz der Deep-AD-Literatur [18] |
| **Sequenzbasiert (fensterweise)** | DPCA (Lag-Features) [5], LSTM-Autoencoder [20] | Adressieren D3 (Dynamik, Drift-Fehler) bei kompatibler Run-Struktur (D4, fensterweise pro Lauf) |
| **Supervised (nur als Obergrenze)** | RandomForest, Gradient Boosting | D6 erlaubt es; aber Vorsicht: supervised Modelle erkennen nur *bekannte* Fehlertypen – reale Anomalien sind per Definition neu. Daher Referenz-Obergrenze, nicht Hauptlinie [16] |

### 3.2 Ungeeignet oder nachrangig (mit Grund)

| Familie | Grund des Ausschlusses |
|---|---|
| **Univariate Regelkarten (Shewhart, EWMA, CUSUM je Sensor)** | Ignorieren D2; dienen nur als Negativ-Referenz [3] |
| **Matrix-Profile / Discord-Discovery** [15] | Setzen *eine* lange Serie voraus, suchen punktuelle/kollektive Einzelmuster – inkompatibel mit D4 (500 unabhängige Läufe) und D9 (persistente Zustandswechsel) |
| **Multimode-/adaptive Monitoring-Verfahren** | Lösen ein Problem (Betriebspunktwechsel, Drift der Normalverteilung), das laut D8 nicht vorliegt |
| **Graph-/kategoriale AD-Verfahren** | Datentyp passt nicht (D1: rein numerisch-tabular) |
| **Forecasting-basierte AD auf Einzelserien (ARIMA-Residuen u. ä.)** | Univariat bzw. schlecht auf 52 gekoppelte Kanäle und 500 kurze Läufe übertragbar; multivariate Alternativen (DPCA, LSTM-AE) decken den Zweck besser ab |

---

## 4. Eingrenzung der AutoML-Frameworks

### 4.1 Das Grundproblem

Klassisches AutoML (CASH: *Combined Algorithm Selection and Hyperparameter
Optimization* [21, 22]) braucht eine **Validierungsmetrik mit Labels**. Für
unsupervised Anomalieerkennung existiert keine zuverlässige interne Metrik;
Modellselektion ohne Labels ist ein offenes Forschungsproblem [13, 14].
Konsequenz für die Framework-Wahl:

1. **Supervised-AutoML-Frameworks** (auto-sklearn, AutoGluon, FLAML, H2O,
   TPOT) können die Anomalieerkennung selbst **nicht** abbilden – sie kennen
   kein One-Class-Lernen. Sie sind nur für die supervised Obergrenze
   (Fehlerklassifikation) nutzbar.
2. **AD-spezifische „AutoML"-Ansätze** sind entweder Meta-Learning ohne Labels
   (MetaOD [13] – hier unnötig, da Labels vorhanden) oder Benchmark-Harnesses
   über Detektor-Bibliotheken mit labelbasierter Selektion – das ist dank D6
   der methodisch saubere Weg.

### 4.2 Bewertung der Kandidaten

| Framework | Kategorie | Eignung | Begründung |
|---|---|---|---|
| **PyOD** [12] + eigener Selektions-Harness | AD-Bibliothek (>40 Detektoren, einheitliche API) | ✅ **gewählt** | Deckt alle geeigneten Familien aus §3.1 ab; CASH wird durch labelbasierte Selektion (D6) realisiert; skaliert (D7); läuft im modernen Stack |
| **PyCaret** (`anomaly`) | AutoML-Wrapper um PyOD | ⚠️ konzeptionell passend, **technisch ausgeschlossen** | Nicht installierbar unter Python 3.13 (`htmlmin`→`cgi` entfernt); erzwingt pandas<2.2/sklearn<1.6; bietet gegenüber direktem PyOD keine zusätzlichen Modelle (Details: [ANSATZ.md](ANSATZ.md) §3) |
| **TODS** [23] | AutoML für Zeitreihen-AD | ⚠️ nachrangig | Adressiert genau D3, aber Forschungsprototyp mit schwerer Dependency-Kette und auf Einzelserien-Pipelines ausgelegt (Konflikt mit D4) |
| **Merlion** [24] | TS-Bibliothek mit AutoML-Anteilen | ⚠️ nachrangig | Fokus univariates Forecasting/AD; multivariate One-Class-Unterstützung schmaler als PyOD |
| **MetaOD** [13] | Meta-Learning zur unsupervised Modellselektion | ❌ unnötig | Löst Selektion *ohne* Labels – D6 macht das obsolet; labelbasierte Selektion ist strikt stärker |
| **FLAML** [25] | Supervised AutoML (leichtgewichtig) | ✅ für die supervised Obergrenze | Läuft unter Windows im modernen sklearn-Stack; budgetbasiert, gut für D7 |
| **AutoGluon** [26] | Supervised AutoML (Ensembles) | ⚠️ Alternative zur Obergrenze | Stark, aber schwergewichtig; Mehrwert ggü. FLAML für eine reine Referenz-Obergrenze fraglich |
| **auto-sklearn** [22] | Supervised AutoML | ❌ ausgeschlossen | Offiziell nur Linux; pinnt altes scikit-learn – inkompatibel mit Env und OS |
| **H2O AutoML** [27] | Supervised AutoML (JVM) | ❌ ausgeschlossen | JVM-Abhängigkeit ohne fachlichen Mehrwert; kein One-Class-Lernen |
| **TPOT** [28] | Supervised AutoML (genetisch) | ❌ ausgeschlossen | Evolutionäre Pipeline-Suche unverhältnismäßig teuer bei D7; kein One-Class-Lernen |

### 4.3 Ergebnis der Eingrenzung

> **Hauptlinie (Anomalieerkennung):** PyOD-Detektoren der Familien aus §3.1,
> orchestriert durch einen Selektions-Harness, der dank vorhandener Labels (D6)
> Modell- und Schwellenwahl automatisch über eine Validierungsmetrik (PR-AUC)
> trifft – das CASH-Prinzip [21], angewandt auf One-Class-Detektoren.
> Als Referenz davor: MSPC-Baseline (PCA-T²/SPE) als etablierter TEP-Standard [4, 6].
>
> **Nebenlinie (Obergrenze):** FLAML für die supervised Fehlerklassifikation,
> klar als Obergrenze gekennzeichnet, weil supervised Modelle nur bekannte
> Fehlertypen erkennen [16].

---

## Quellen

[1] J. J. Downs, E. F. Vogel: *A plant-wide industrial process control problem.* Computers & Chemical Engineering 17(3), 245–255, 1993.

[2] C. A. Rieth, B. D. Amsel, R. Tran, M. B. Cook: *Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation.* Harvard Dataverse, 2017. doi:10.7910/DVN/6C3JR1

[3] L. H. Chiang, E. L. Russell, R. D. Braatz: *Fault Detection and Diagnosis in Industrial Systems.* Springer, 2001.

[4] E. L. Russell, L. H. Chiang, R. D. Braatz: *Fault detection in industrial processes using canonical variate analysis and dynamic principal component analysis.* Chemometrics and Intelligent Laboratory Systems 51(1), 81–93, 2000.

[5] W. Ku, R. H. Storer, C. Georgakis: *Disturbance detection and isolation by dynamic principal component analysis.* Chemometrics and Intelligent Laboratory Systems 30(1), 179–196, 1995.

[6] S. Yin, S. X. Ding, A. Haghani, H. Hao, P. Zhang: *A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process.* Journal of Process Control 22(9), 1567–1581, 2012.

[7] B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, R. C. Williamson: *Estimating the support of a high-dimensional distribution.* Neural Computation 13(7), 1443–1471, 2001.

[8] F. T. Liu, K. M. Ting, Z.-H. Zhou: *Isolation Forest.* IEEE International Conference on Data Mining (ICDM), 2008.

[9] Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu, G. H. Chen: *ECOD: Unsupervised outlier detection using empirical cumulative distribution functions.* IEEE Transactions on Knowledge and Data Engineering, 2022.

[10] M. M. Breunig, H.-P. Kriegel, R. T. Ng, J. Sander: *LOF: Identifying density-based local outliers.* ACM SIGMOD, 2000.

[11] M. Goldstein, S. Uchida: *A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data.* PLOS ONE 11(4), 2016.

[12] Y. Zhao, Z. Nasrullah, Z. Li: *PyOD: A Python toolbox for scalable outlier detection.* Journal of Machine Learning Research 20(96), 1–7, 2019.

[13] Y. Zhao, R. A. Rossi, L. Akoglu: *Automatic unsupervised outlier model selection.* NeurIPS, 2021.

[14] M. Q. Ma, Y. Zhao, X. Zhang, L. Akoglu: *The need for unsupervised outlier model selection: A review and evaluation of internal evaluation strategies.* ACM SIGKDD Explorations 25(1), 2023.

[15] C.-C. M. Yeh et al.: *Matrix Profile I: All pairs similarity joins for time series.* IEEE ICDM, 2016.

[16] V. Chandola, A. Banerjee, V. Kumar: *Anomaly detection: A survey.* ACM Computing Surveys 41(3), 2009.

[17] R. Wu, E. Keogh: *Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress.* IEEE Transactions on Knowledge and Data Engineering 35(3), 2023.

[18] L. Ruff et al.: *A unifying review of deep and shallow anomaly detection.* Proceedings of the IEEE 109(5), 756–795, 2021.

[19] J. An, S. Cho: *Variational autoencoder based anomaly detection using reconstruction probability.* SNU Data Mining Center, Technical Report, 2015.

[20] P. Malhotra et al.: *LSTM-based encoder-decoder for multi-sensor anomaly detection.* ICML Anomaly Detection Workshop, 2016.

[21] C. Thornton, F. Hutter, H. H. Hoos, K. Leyton-Brown: *Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms.* ACM KDD, 2013.

[22] M. Feurer, A. Klein, K. Eggensperger, J. T. Springenberg, M. Blum, F. Hutter: *Efficient and robust automated machine learning.* NeurIPS, 2015.

[23] K.-H. Lai, D. Zha, et al.: *TODS: An automated time series outlier detection system.* AAAI (Demo), 2021.

[24] A. Bhatnagar et al.: *Merlion: A machine learning library for time series.* arXiv:2109.09265, 2021.

[25] C. Wang, Q. Wu, M. Weimer, E. Zhu: *FLAML: A fast and lightweight AutoML library.* MLSys, 2021.

[26] N. Erickson et al.: *AutoGluon-Tabular: Robust and accurate AutoML for structured data.* arXiv:2003.06505, 2020.

[27] E. LeDell, S. Poirier: *H2O AutoML: Scalable automatic machine learning.* ICML AutoML Workshop, 2020.

[28] R. S. Olson, J. H. Moore: *TPOT: A tree-based pipeline optimization tool for automating machine learning.* ICML AutoML Workshop, 2016.

> *Hinweis: Die Quellen sind etablierte Standardliteratur der jeweiligen
> Teilgebiete; DOIs/Links bei Bedarf für die Abgabe ergänzen und prüfen.*
