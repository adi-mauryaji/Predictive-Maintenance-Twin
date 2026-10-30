# Predictive Maintenance Twin · Project Guide

Ye guide project ko **samajhne** ke liye hai: problem, data, ML ka logic, trade-offs, architecture aur build plan.
Frozen spec (folder layout, API contract, pinned versions, har milestone ka exit criteria) KYROS vault mein hai:
`Projects/predictive-maintenance-twin/ROADMAP.md` (ADR-002). Dono mein farq dikhe to **spec jeetega**, aur guide theek hogi.

Dataset ke saare numbers 2026-10-30 ko asli files se nikaale gaye hain (model ki yaad se nahi).

---

## 1. Ek nazar mein

> Doctor ka regular checkup: har visit par vitals naapta hai, "normal" se compare karta hai, aur heart attack se pehle warning deta hai.

Hum ek **turbofan jet engine fleet** ka digital twin banayenge. Har engine ke liye ek twin:

1. har cycle (ek flight) ki sensor readings leta hai,
2. pakadta hai ki engine "normal" se hat raha hai ya nahi (**anomaly / health**),
3. batata hai ki failure tak lagbhag **kitne cycles bache** hain (**RUL**, Remaining Useful Life), ek range ke saath,
4. aur status deta hai: HEALTHY / WATCH / CRITICAL + kya karna chahiye.

| Doctor wali duniya | Hamara system |
|---|---|
| Patient | Ek engine |
| Vitals (BP, pulse, sugar) | 21 sensors + 3 operating settings |
| "Normal range" ka gyaan | Health model, jo engines ki shuruaati (healthy) life par train hua |
| ECG mein kuch ajeeb | Anomaly alarm |
| "Aise chala to itna time bacha hai" | RUL prediction + uncertainty range |
| Medical file | Twin ki history (SQLite) |
| Prescription | Maintenance recommendation |
| Regular checkup | Har cycle ki reading ka ingestion |

Flow: **sensors (replay simulator) → FastAPI twin service → anomaly + RUL models → Streamlit dashboard**.

---

## 2. Problem: maintenance ke teen tareeke

| Tareeka | Kab theek karte hain | Nuksaan |
|---|---|---|
| Reactive (run-to-failure) | Tootne ke baad | Achanak downtime, safety risk, mehenga emergency repair |
| Preventive (time-based) | Fixed schedule par (har N ghante) | Healthy parts bhi badal dete hain; schedule ke beech mein fail ho sakta hai |
| **Predictive (condition-based)** | Jab data bataye ki failure paas hai | Data, models aur unki galtiyon ko sambhalna padta hai |

Predictive maintenance part ki poori life use karta hai aur failure se pehle action leta hai.
Hum ek chhota **prescriptive** hissa bhi jodenge: sirf "kab" nahi, "kya karo" bhi (status rules se recommendation).

---

## 3. Digital twin kya hai (aur kya nahi)

| Level | Matlab | Data ka flow |
|---|---|---|
| Digital model | Offline model, asset se koi live link nahi | Koi nahi |
| Digital shadow | Asset ka data apne aap digital copy mein aata hai | Physical → digital |
| **Digital twin** | Shadow + digital side se decision/action wapas asset tak | Dono taraf |

Ek twin ke teen hisse hote hain: **state** (abhi asset kaisa hai), **models** (physics ya ML), **history** (pehle kya hua).

Hamara twin: live replay se state sync hoti hai, models anomaly aur RUL nikaalte hain, history SQLite mein rehti hai.
Wapas wala hissa: recommendation, aur `POST /engines/{id}/maintenance`, jo "maintenance ho gaya" record karke twin ko nayi life se shuru karta hai.

Imaandari se: asli engine nahi hai, ek **simulated** engine ka twin hai. Sensors ki jagah simulator NASA ke data ko cycle-by-cycle "replay" karta hai.
Architecture aisi hai ki kal asli sensors (MQTT ke through) aa jaayein to sirf ingestion badlegi, twin nahi.

---

## 4. Dataset: NASA C-MAPSS

**C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) NASA ka turbofan engine simulator hai.
Data ek fleet jaisa hai: har engine thodi alag manufacturing variation aur shuruaati wear ke saath start hota hai (jo "normal" maana gaya hai).
Shuru mein engine normal chalta hai, beech mein kahin ek fault shuru hota hai, aur fault badhta jaata hai jab tak engine fail na ho. Readings mein sensor noise mila hua hai.

### 4.1 Chaar subsets (asli files se verified)

| Subset | Train engines | Test engines | Operating conditions | Fault modes | Train life (min / median / max cycles) |
|---|---|---|---|---|---|
| **FD001** | 100 | 100 | 1 (sea level) | 1 (HPC degradation) | 128 / 199 / 362 |
| FD002 | 260 | 259 | 6 | 1 (HPC) | 128 / 199 / 378 |
| FD003 | 100 | 100 | 1 | 2 (HPC + Fan) | 145 / 220 / 525 |
| FD004 | 249 | 248 | 6 | 2 (HPC + Fan) | 128 / 234 / 543 |

Quirk: FD004 ka readme "248 train / 249 test" bolta hai, lekin files mein 249 train / 248 test engines hain. Lesson: **readme par nahi, data par bharosa karo**.

HPC = High Pressure Compressor. Fan = engine ka aage wala fan.

### 4.2 Files aur columns

Har file mein space-separated 26 columns: `unit, cycle, setting1..3, s1..s21`. Ek row = ek engine ki ek cycle (ek flight) ka snapshot.

```
train_FD001.txt  : har engine shuru se FAILURE tak (run-to-failure)    20,631 rows
test_FD001.txt   : har engine failure se PEHLE kahin kaat diya gaya     13,096 rows  (length 31..303 cycles)
RUL_FD001.txt    : test ke har engine ki aakhri cycle ke baad asli bache cycles   100 values (7..145)

engine 1 (train):  cycle 1 ─────────────────────────────── cycle 192 ✖ fail
engine 1 (test) :  cycle 1 ───────────── cycle 31 |  ? (asli jawab RUL_FD001 mein: 112)
```

**Unit:** RUL **cycles** mein hai (flights), seconds mein nahi. Simulator mein hum "1 cycle = X seconds" replay speed rakhenge, sirf demo ke liye.

### 4.3 FD001 ke sensors: kaun kaam ke hain

Train data par har column ka standard deviation (sd) aur RUL ke saath correlation (r):

- **Bilkul constant (sd = 0):** setting3, s1, s5, s10, s16, s18, s19 → koi information nahi, hata do.
- **Lagbhag constant:** s6 (sd 0.0014, r −0.13), setting1 aur setting2 (sirf noise) → hata do.
- **Kaam ke 14 sensors:** s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.
- Sabse strong: s11 (r −0.70), s4 (−0.68), s12 (+0.67), s7 (+0.66), s15 (−0.64), s21 (+0.64), s20 (+0.63).

Correlation sirf linear rishta naapta hai aur ek engine ki trend ko poori fleet ke saath mila deta hai. EDA (M1) mein har sensor ko RUL ke against plot karke dekhenge.

### 4.4 FD003 kyun interesting hai

FD001 mein s20 aur s21 ka RUL ke saath r ≈ +0.63; FD003 (do fault modes) mein wahi r ≈ +0.04 aur +0.03 ho jaata hai.
Matlab doosra fault mode **alag sensors** hilata hai. Ek FD001 par train model FD003 par galat confidence ke saath galat bolega.
Hum isko M8 mein **out-of-distribution demo** banayenge: FD003 replay karo, aur dekho drift monitor aur anomaly model pakadte hain ya nahi.

### 4.5 Data kahan se milega

- data.nasa.gov ki official C-MAPSS listing abhi "currently unavailable for download" bolti hai.
- PHM Society ka NASA repository mirror chal raha hai (2026-10-30 ko HTTP 200, 12,429,152 bytes).
- Download script us zip ka sha256 check karegi: `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`. Andar ek aur zip (`CMAPSSData.zip`) hai.
- Citation: A. Saxena, K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository. License likha nahi hai; repository "use at own risk" bolti hai aur acknowledgement maangti hai. README mein citation denge.
- N-CMAPSS (2021, asli flight conditions) 15.76 GB hai: is laptop ke liye bahar.

---

## 5. Do ML problems

### 5.1 Anomaly / health: "normal kaisa dikhta hai"

C-MAPSS mein "yahan anomaly hai" jaisa koi label nahi hai. Isliye hum **normal** seekhte hain aur usse doori naapte hain.

1. **Healthy data:** har fit engine ki pehli 50 cycles (sabse chhoti life bhi 128 hai, to ye shuruaati hissa safe hai).
2. **PCA** se 14 sensors ka "normal pattern" seekho. Phir har nayi reading ke liye do score:
   - **T² (Hotelling):** reading normal pattern ke andar hai, par center se kitni door?
   - **SPE / Q:** reading normal pattern se hi bahar kitni hai (sensors ka aapsi rishta toota)?
3. **Threshold:** validation engines ki healthy cycles (1–50) par score ka 99th percentile.
4. **Persistence:** alarm tab, jab last 5 cycles mein se 3 threshold ke paar hon. Ek noisy reading par alarm nahi.
5. **Explain:** SPE mein har sensor ka hissa → top 3 sensors dashboard par ("s11 aur s4 hil rahe hain").
6. **Challenger:** IsolationForest (scikit-learn). Jo validation par better nikle wahi champion.

Evaluation (label nahi hain, to proxy metrics):
- **False alarm rate:** healthy validation cycles par kitne alarm.
- **Lead time:** pehle persistent alarm se failure tak kitne cycles (median).
- **Coverage:** kitne % engines par alarm RUL ≤ 30 se pehle aaya.

### 5.2 RUL: kitne cycles bache

Train data mein label banana aasaan hai: `RUL = engine ki aakhri cycle − current cycle`.

Problem: shuruaati life mein engine theek hai, degradation dikhti hi nahi. Model ke paas koi tareeka nahi ki 300 aur 200 bache cycles mein farq kare.
Isliye standard **piecewise-linear target**: RUL ko 125 par cap karo.

```
RUL label
 125 ┤━━━━━━━━━━━━━━━━━━━┓            capped: "abhi healthy hai"
     │                    ┃╲
     │                    ┃  ╲          linear: har cycle 1 kam
     │                    ┃    ╲
   0 ┤────────────────────┸──────╲──▶ cycle
                               failure
```

Trade-off: model 125 se zyada kabhi nahi bolega. Theek hai, kyunki maintenance ka decision failure ke paas hi hota hai.

---

## 6. Feature engineering (time-series ka asli kaam)

Ek single reading noisy hoti hai. **Trend** (kis taraf ja raha hai) signal hai.

| Step | Kya | Kyun |
|---|---|---|
| Sensor selection | sd ≈ 0 wale hatao (section 4.3), train data se compute | Bekaar columns sirf noise aur cost badhaate hain |
| Scaling | StandardScaler, **sirf training fold par fit** | Sensors alag scale par hain (FD001 mean: s9 ≈ 9065, s4 ≈ 1409, s15 ≈ 8.4) |
| Rolling stats | Last 5 / 10 / 20 cycles ka mean, std | Noise smooth hota hai |
| Slope | Last 10 / 20 cycles par linear regression ka slope | "Kitni tezi se bigad raha hai" |
| EWMA | Exponentially weighted mean | Nayi readings ko zyada weight |
| Age | Current cycle number | Test mein bhi pata hota hai, legit feature |
| Sequences (LSTM) | Last W = 30 cycles × 14 sensors ki window | LSTM khud trend seekhta hai; FD001 test ka sabse chhota engine 31 cycles ka hai, to W = 30 sab par fit |
| Regime normalization (FD002/4, stretch) | 6 operating conditions ko cluster karke har condition mein alag scaling | Warna model "altitude" seekhega, degradation nahi |

### Leakage: sabse common galti

- **Split engine se karo, row se nahi.** Ek engine ki cycles train aur validation dono mein gayi to model "is engine ko yaad" kar lega aur metrics jhoothe achhe aayenge. Test isko pakdega (no engine in both splits).
- Scaler, thresholds, conformal calibration: sab sirf un engines par fit jo evaluation mein nahi hain.
- Official test set har final model par **ek hi baar** chalega. Uspe tuning nahi.

Split plan (100 train engines, seed 42):

| Hissa | Engines | Kaam |
|---|---|---|
| **fit** | 70 | Models train; LightGBM tuning GroupKFold (5 folds) se, isi ke andar |
| **validation** | 15 | Early stopping, champion choice, anomaly thresholds, status thresholds |
| **calibration** | 15 | Sirf conformal intervals. Kisi aur decision mein use nahi, warna interval ki guarantee kamzor |
| official test | 100 (alag file) | Har final model par ek baar, final report ke liye |

---

## 7. Model ladder

| Model | Input | Fayda | Keemat |
|---|---|---|---|
| Baseline: linear regression | Current cycle ke scaled sensors | Neeche ki line: isse kharab to kuch galat hai | Trend nahi dekhta |
| **LightGBM** | Rolling features (section 6) | Tabular data par bahut common, jaldi train, har prediction ki explanation (`pred_contrib` = SHAP values, bina extra library) | Features khud banane padte hain |
| **LSTM** (PyTorch) | 30 × 14 sequence | Trend khud seekhta hai | CPU par minutes mein train, kam explainable, 30 cycles se pehle prediction nahi |

**Champion rule (frozen):** validation engines par RMSE + NASA score dekh ke champion chuna jayega. Barabari ho to LightGBM (sasta, explainable).
LSTM ko apni jagah kamaani padegi. Ye khud ek trade-off lesson hai: "deep learning hamesha jeetega" sach nahi hai.

Twin cold start: engine ke pehle 30 cycles mein LSTM ke paas window nahi hoti, to wahan LightGBM bolega.

### Kya umeed rakhein (imaandari se)

FD001 par published RMSE zyaadatar 11 se 18 ke beech milte hain (kuch papers ~8 ya ~31 bhi report karte hain), lekin papers alag preprocessing, RUL cap aur splits use karte hain, to unko seedha compare nahi kar sakte.
Hamara target koi magic number nahi hai. Target hai: baseline < LightGBM < (shayad) LSTM, sab ek hi protocol par, aur numbers jo bhi aayein woh sach likhe jayenge.

---

## 8. Metrics

- **RMSE** (bade errors ko zyada saza), **MAE** (average galti, cycles mein).
- **NASA score** (PHM08 challenge), `d = predicted − true`:
  - `d < 0` (early, pehle bola): `e^(−d/13) − 1`
  - `d ≥ 0` (late, der se bola): `e^(d/10) − 1`
  - Example: 10 cycles late = 1.72, 10 cycles early = 1.16. 30 late = 19.1, 30 early = 9.0.
  - Matlab: **der se bolna zyada khatarnaak** hai (engine pehle hi fail ho jayega). Model thoda conservative hona chahiye.
- **Late-life RMSE:** sirf jahan true RUL ≤ 30. Decisions yahin hote hain.
- **Interval coverage aur width:** 90% interval mein sach kitni baar aaya, aur range kitni chaudi hai.
- Anomaly metrics: section 5.1.

---

## 9. Uncertainty: conformal prediction

Ek number ("42 cycles bache") se zyada kaam ki baat: "**30 se 55 ke beech, 90% bharose ke saath**".

Split conformal ka idea:
1. Model ko fit engines par train karo.
2. Calibration engines (jinhe model ne kabhi nahi dekha, aur jin par koi aur decision nahi hua) par uski galtiyan `|true − pred|` dekho.
3. Galtiyon ka 90th percentile `q` nikaalo. Interval = `pred ± q`.

**Mondrian** version: predicted RUL ke bins (jaise < 50, 50–100, > 100) ke liye alag `q`, kyunki failure ke paas galti chhoti hoti hai aur shuru mein badi.
Ye ~20 lines ka apna code hai, koi library nahi (MAPIE jaisi library ki zaroorat nahi).

Shart: guarantee tabhi hai jab future data calibration data jaisa ho. FD003 jaisa naya fault mode aaya to interval jhootha ho sakta hai. Isliye drift monitor zaroori hai.

---

## 10. Trade-offs (is project ka asli sabak)

| Decision | Option A | Option B | Hamara choice | Kyun |
|---|---|---|---|---|
| RUL target | Raw linear | Cap 125 | Cap 125 | Shuruaati life mein signal nahi hai; decisions end ke paas |
| Early vs late galti | Symmetric loss (MSE) | Late ko zyada saza | MSE + NASA score se judge, asymmetric loss ek experiment | Late prediction khatarnaak; dekhna hai RMSE kitna kharab hota hai |
| Window W | Chhoti (10) | Lambi (50) | 30 | Lambi = zyada context par cold start der se aur kam training windows |
| Model | LightGBM | LSTM | Data se decide (champion rule) | Accuracy vs training cost, explainability, serving size |
| Anomaly threshold | Neecha | Ooncha | 99th percentile + 3-of-5 | False alarm vs der se pakadna; persistence 2–3 cycle der laata hai par noise kaatta hai |
| Prediction | Single number | Interval | Dono | Maintenance planner ko risk chahiye, sirf average nahi |
| Validation split | Row-wise | Engine-wise | Engine-wise | Row-wise leakage se metrics jhoothe achhe aate hain |
| Serving | PyTorch model | ONNX Runtime | ONNX | Serving image mein bhaari torch nahi chahiye; framework-independent inference |

Ek aur bada sabak: C-MAPSS **simulated** hai. Asli sensors mein missing values, sensor drift, irregular sampling, maintenance ke beech mein records hote hain. Hamare results simulated data ke results hain, aur README mein yahi likhenge.

---

## 11. Architecture

```
 OFFLINE · training (laptop CPU)                                      artifacts/
 ┌────────────┐   ┌──────────────────┐   ┌──────────────────────┐    health.joblib
 │ data/raw   │──▶│ features         │──▶│ train + evaluate     │──▶ gbm.joblib
 │ C-MAPSS    │   │ select, scale,   │   │ health, baseline,    │    lstm.onnx
 │ (download, │   │ rolling, windows │   │ LightGBM, LSTM,      │    conformal.json
 │  sha256)   │   └──────────────────┘   │ conformal (MLflow)   │    model_card.json
 └────────────┘                          └──────────────────────┘          │
                                                                           │ load at startup
 ONLINE · serving                                                          ▼
 ┌────────────┐  POST /ingest     ┌──────────────────────────────────────────────────┐
 │ replay     │──────────────────▶│ FastAPI twin service                              │
 │ simulator  │  readings (JSON)  │  TwinRegistry ── EngineTwin × N                   │
 │ test_FD00x │                   │    window buffer (30) → health model → RUL model │
 └────────────┘                   │    → conformal interval → status rules           │
                                  │  SQLite: readings · predictions · events         │
                                  └─────────────────────┬────────────────────────────┘
                                                        │ GET /fleet · /engines/{id} · /models
                                  ┌─────────────────────▼────────────┐
                                  │ Streamlit dashboard               │
                                  │ Fleet · Engine detail · Models    │
                                  └───────────────────────────────────┘
```

### Ek reading ka safar

1. Simulator `test_FD001` ke engine 7 ki cycle 45 bhejta hai: `POST /ingest`.
2. FastAPI Pydantic se validate karta hai (galat sensor naam ya missing value → 422; purani cycle dobara → 409).
3. `EngineTwin(7)` reading ko apne 30-cycle buffer mein daalta hai.
4. Health model T² / SPE score deta hai; persistence rule alarm decide karta hai.
5. RUL model (champion; buffer chhota ho to LightGBM) point prediction deta hai; conformal se interval.
6. Status rules: HEALTHY / WATCH / CRITICAL + recommendation.
7. Reading aur snapshot SQLite mein. Response mein naya snapshot.
8. Dashboard har kuch second mein `GET /fleet` se fleet ka haal dikhata hai.

### Boundaries (frozen rules)

- **Dashboard sirf HTTP API se baat karta hai.** Models ya SQLite seedha nahi chhoota. UI badle to twin nahi tootta.
- **Serving torch import nahi karta.** LSTM ONNX mein export hota hai, onnxruntime chalata hai. Export ke baad torch aur ONNX outputs ka match test hota hai.
- **Twin ko transport nahi pata.** `ingest()` ek function hai; HTTP route aaj usse bulata hai, kal MQTT subscriber bulayega.
- **Models ek interface ke peeche.** Tests mein fake predictors, asli models ke bina API test hoti hai.
- **Twin restart-safe:** service restart par SQLite se har engine ki last 30 readings padh ke state dobara banti hai.

### API (summary; exact contract vault ROADMAP mein)

| Method | Path | Kaam |
|---|---|---|
| GET | `/health` | Service + loaded model versions |
| POST | `/ingest` | Ek ya zyada readings → updated snapshots |
| GET | `/fleet` | Har engine: status, RUL + range, anomaly score, alarm |
| GET | `/engines/{id}` | Ek engine ka latest snapshot (top sensors, recommendation) |
| GET | `/engines/{id}/history` | Readings + predictions over time |
| POST | `/engines/{id}/maintenance` | Maintenance record, twin nayi life se shuru (two-way action) |
| POST | `/predict` | Stateless: ek window do, RUL lo (twin ke bina model test) |
| GET | `/models` | Model cards: metrics, version, train date, data hash |
| GET | `/monitoring/drift` | Har sensor ka PSI: training vs recent readings |

### Stack: kaunsa tool, kyun

| Kaam | Tool | Kyun ye | Kya nahi liya, kyun |
|---|---|---|---|
| Env, deps | uv + `uv.lock` | Fast, exact reproducible versions | pip + requirements.txt: lock kamzor |
| Data | pandas + Parquet (pyarrow) | Standard, processed data fast load | Polars: achha hai par ek naya API seekhna padta |
| Classical ML | scikit-learn (PCA, IsolationForest, pipelines, GroupKFold) | Industry standard | — |
| GBM | LightGBM | Tabular data par bahut common, built-in SHAP values | XGBoost: barabar, ek hi kaafi |
| Deep learning | PyTorch (CPU) | Training loop khud likhna padta hai (seekhne ke liye achha), ONNX export built-in | Keras/TF: barabar kaam karta, ek framework kaafi; Lightning: pehle plain loop samjho |
| Serving inference | ONNX Runtime | Chhota, framework-independent | torch in serving: bhaari install |
| API | FastAPI + Pydantic v2 | Typed validation, auto docs (`/docs`) | Flask: validation khud likhna |
| Storage | SQLite (stdlib) | Zero setup, 100 engines × ~300 cycles ke liye kaafi | TimescaleDB: stretch S3 |
| Dashboard | Streamlit + Plotly | Python mein jaldi interactive UI | Grafana: ops dashboards ka common tool, stretch S3 |
| Experiment tracking | MLflow (local file store) | MLOps ka standard tool, ₹0 local | W&B: paid cloud |
| Drift | Apna PSI code | ~15 lines | Evidently: bhaari dependency |
| Tests, lint | pytest, ruff | Standard | — |
| Packaging | Docker Compose | Ek command mein poora system | Kubernetes: is scale par overkill |

Budget: **₹0**. Sab open-source, sab local. Koi cloud, koi paid API nahi.

---

## 12. Market se rishta

Ye project kin skills ko dikhata hai, aur unka saboot kahan se aaya:

| Skill | Project mein kahan | Evidence (2026-10-30 ki search) |
|---|---|---|
| Anomaly detection, time-series forecasting, classical + deep models | M3, M4, M5 | AssetWatch "ML Engineer, Predictive Maintenance" posting: anomaly detection, time-series forecasting, Docker, SQL, cloud |
| Model API + continuous monitoring (MLOps) | M6, M8 | 2026 Coursera course "AI in Manufacturing: Predictive Maintenance and Alerts": MLOps, monitoring, model API |
| Uncertainty / risk-aware RUL | M4 (conformal) | Conformal RUL papers 2023–2026 (Javanmardi & Hüllermeier 2023; Moccardi et al. 2025; Robinson 2026), sab C-MAPSS par |
| IIoT data pipeline (MQTT → time-series DB → Grafana) | Stretch S2, S3 | United Manufacturing Hub aur EMQX Open Manufacturing Hub reference architectures |
| Reproducible env, Docker, tests | M0, M9 | AssetWatch posting (Docker), general MLOps demand |

Imaandari: ye evidence patla hai (kuch postings, ek course, kuch papers, do reference architectures). Ye formal job-market survey nahi hai.
Isliye plan ka core (leakage-free time-series ML, honest model comparison, uncertainty, API + dashboard, Docker) aisa rakha hai jo kisi bhi ML/MLOps role mein kaam aaye. Industry-specific cheezein (MQTT, TimescaleDB, Grafana) stretch mein hain.

---

## 13. Build plan

Kaise sikhaya jayega: `~/ai-engineer/CLAUDE.md` ke "How to teach" rules (pehle design MCQ round, phir setup ki har command samjhayi jayegi, phir chhote chunks; har chunk ke baad explanation + 3–5 MCQs, aur chunk run karke output dekhna). Learning state: `~/Digital Twinning/learning/`. Faisla: KYROS ADR-004.
Bhaari kaam (torch download, LSTM training) se pehle time / RAM / CPU bata ke permission li jayegi.

| # | Milestone | Kya banega | Kya seekhoge |
|---|---|---|---|
| M0 | Setup | uv project, deps, ruff, pytest, download script (sha256) | Reproducible Python env, data provenance |
| M1 | Data + EDA | Loaders, RUL labels, EDA notebook, sensor selection | Run-to-failure data padhna, EDA, label design |
| M2 | Features + splits | Scaling, rolling, slope, windows, engine-wise split + leakage test | Time-series feature engineering, leakage |
| M3 | Health / anomaly | PCA T² + SPE, IsolationForest challenger, thresholds, persistence | Unsupervised anomaly detection, threshold trade-off |
| M4 | RUL classical | Baseline, LightGBM, metrics, NASA score, conformal, MLflow | GBM on time series, metrics, uncertainty, tracking |
| M5 | RUL deep | LSTM (PyTorch), ONNX export + parity test, champion choice | Sequence models, CPU training, model export |
| M6 | Twin service | EngineTwin, registry, status rules, SQLite, FastAPI, tests | Stateful service design, API contracts, testing |
| M7 | Simulator + dashboard | Replay CLI, Streamlit fleet / engine / models pages | Streaming simulation, ML dashboards |
| M8 | Monitoring + robustness | PSI drift, FD003 OOD demo, error cases, maintenance reset | Monitoring, model limits |
| M9 | Packaging | Dockerfile (bina torch), Compose, README with results | Shipping an ML system |
| S1–S5 | Stretch (naye ADR se) | FD002/FD004, MQTT, TimescaleDB + Grafana, CNN/Transformer, drift-triggered retrain | Industry depth |

---

## 14. Ye project kya nahi hai

- Asli hardware, cloud deploy, login/auth, Kubernetes, Kafka.
- Physics-based twin (thermodynamic engine model). Hamara twin data-driven hai.
- N-CMAPSS (15.76 GB) ya GPU training.
- "State of the art" ka daawa. Hamare numbers sirf hamare protocol par valid hain.

---

## 15. Sources

- C-MAPSS data (mirror): https://data.phmsociety.org/nasa/ (zip: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)
- Official listing (abhi unavailable): https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data
- Saxena, Goebel, Simon, Eklund (2008), "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08 (zip ke andar PDF).
- AssetWatch posting: https://talents.vaia.com/?p=11180391
- Coursera, AI in Manufacturing: Predictive Maintenance and Alerts: https://www-cloudfront-alias.coursera.org/learn/ai-in-manufacturing-predictive-maintenance-and-alerts
- Javanmardi & Hüllermeier (2023), Conformal Prediction Intervals for RUL Estimation, IJPHM: https://papers.phmsociety.org/index.php/ijphm/article/view/3417
- Moccardi et al. (2025), A Robust Conformal Framework for IoT-Based Predictive Maintenance, Future Internet 17(6): https://ideas.repec.org/a/gam/jftint/v17y2025i6p244-d1668249.html
- Robinson (2026), RUL Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression, IJPHM 17(1): https://papers.phmsociety.org/index.php/ijphm/article/view/4724
- FD001 RUL results ki range (protocols alag): https://arxiv.org/pdf/2603.00745, https://scholarworks.unist.ac.kr/handle/201301/83174
- United Manufacturing Hub (MQTT + TimescaleDB + Grafana): https://www.tigerdata.com/blog/how-united-manufacturing-hub-is-introducing-open-source-to-manufacturing-and-using-time-series-data-for-predictive-maintenance
- EMQX Open Manufacturing Hub: https://dev.to/emqx/open-manufacturing-hub-a-reference-architecture-for-industrial-iot-iiot-2h9p
