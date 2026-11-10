# Instructor guide

8 h practical, four 2-hour sessions. Everything below was run on the two CSVs in
`data/` with scikit-learn 1.9.1 / pandas 3.0.6 / Python 3.11; every number
quoted in the notebooks and in this guide came out of that run. Seeds are fixed
(`RNG = 0`), so students should reproduce them to the third decimal. Minor
scikit-learn version differences can move the forest results by ~0.005.

---

## The design choice behind the course

The models in this practical are deliberately unambitious: linear regression,
ridge, k-nearest neighbours, a random forest, gradient boosting. Nothing after
session 1 is harder to fit than two lines of scikit-learn. The entire
difficulty budget is spent on *measurement* — baselines, resampling, leakage,
selection bias, applicability domain.

The reason is that a quantitative-biology graduate will not be out-engineered on
model architecture; they will be caught out by a result that does not replicate.
The failure modes in sessions 2-4 are the ones that appear in real manuscripts:
feature selection outside the resampling loop, a performance figure from a
single lucky split, a cross-validated score quoted for molecules the model has
no business predicting.

**Three of the exercises produce deliberate nulls.** Scaler leakage (+0.004),
duplicate-row leakage in aggregate (+0.004), and selection bias at n=908
(-0.01) are all too small to detect on this data. This is intentional and it is
the most important thing in the course to protect from being "fixed". Students
who only ever see rigged demonstrations learn that bad practice always shows up
in the headline number, which is false and leaves them unable to defend
methodology when it costs them performance. Each null is paired with a regime
where the same mechanism is large: the selection leak becomes +0.39 at n=60 with
2000 descriptors, the duplicate leak becomes +0.13 when scored on the affected
rows only, selection bias becomes +0.062 at n=120. The lesson is that effect size
depends on the regime, and that you cannot audit someone else's regime from
their abstract.

---

## Timing

| Session | Block | Minutes |
|---|---|---|
| **1** | Intro: QSAR, REACH, what the target means | 15 |
| | §1 data + EX 1.1-1.2 (target, descriptors, correlations) | 30 |
| | §2 baselines + EX 1.3 (why dummy R² is negative) | 20 |
| | §3 one split + EX 1.4 (ten seeds) | 25 |
| | §4 cross-validation + EX 1.5 (fit for purpose, MLOGP alone) | 25 |
| | Wrap-up | 5 |
| **2** | §1 EX 2.1 split lottery (200 seeds, ~1 min compute) | 30 |
| | EX 2.2 repeated K-fold stability | 15 |
| | §2 EX 2.3 scaling per model family | 15 |
| | §3a EX 2.4 scaler leakage — the first null | 15 |
| | §3b EX 2.5-2.6 feature-selection leakage, n small / p large | 30 |
| | §3c EX 2.7 duplicate molecules | 20 |
| | Checklist + homework briefing | 5 |
| **3** | §1 EX 3.1 six-model table with error bars (~2 min compute) | 25 |
| | EX 3.2 paired fold differences; why no p-value | 20 |
| | §2 EX 3.3 validation curve for k | 15 |
| | EX 3.4 the 39-config leaderboard (~3 min compute) | 20 |
| | §3 EX 3.5-3.6 nested CV, n=908 then n=120 (~8 min compute) | 25 |
| | §4 EX 3.7 learning curves + the budget question | 15 |
| **4** | §1 EX 4.1 y-scrambling (~2 min compute) | 20 |
| | §2 EX 4.2 residuals, worst molecules, the descriptor floor | 25 |
| | §3 EX 4.3-4.4 importance, impurity vs permutation, collinearity | 30 |
| | §4 EX 4.5-4.6 extrapolation and a domain indicator | 30 |
| | §5 capstone briefing | 15 |

**If you are running short.** Session 2: cut EX 2.2 and 2.6. Session 3: cut EX
3.6 and demo the result from this guide instead — it is the most expensive cell
in the course. Session 4: cut EX 4.4. Never cut EX 2.5 (feature-selection
leakage) or EX 4.5 (extrapolation); those are the two results students remember.

---

## Session 1 — A first model, and the baseline it has to beat

**Objectives.** Read a dataset in its own units; state a baseline before
modelling; distinguish training from held-out performance; see that a single
split is not an estimate.

**Expected numbers (single 5-fold CV, `random_state=0`, fish).** Note that
session 3 uses 5-fold × 5 repeats and reports marginally different values for the
same models (dummy -0.008, linear 0.560 ± 0.063); both are correct for their
resampling scheme, and the difference is worth pointing out if a student notices.

| predictor | R² | RMSE (log units) | fold-error |
|---|---|---|---|
| dummy (training mean) | -0.016 | 1.456 | 29× |
| `MLOGP` alone, linear | 0.408 | — | — |
| all six, linear | 0.557 ± 0.043 | 0.960 | 9.1× |

Target: mean 4.064, sd 1.455, range 0.053-9.612 (9.56 log units ≈ a factor of
3.6 × 10⁹). Correlations with LC50: MLOGP 0.652, SM1_Dz(Z) 0.411,
GATS1i -0.398, CIC0 0.292, NdsCH 0.172, NdssC 0.172.

Single 75/25 split at `random_state=42`: train R² 0.577, test R² 0.569 — close
together, which is a useful surprise for a linear model with six features.
Ten-seed single-split spread (EX 1.4): **0.442 to 0.618**, spread 0.176.
A factor-of-3 tolerance is 0.477 log units, so the model is 2.0× worse than
that.

**Discussion prompts.**

- The dummy's R² is -0.016, not 0. Getting students to work out why (the
  prediction uses the *training* fold's mean; the R² denominator uses the *test*
  fold's mean) is the first time they have to think about what a held-out score
  actually computes. It pays off in session 4 when the y-scrambling null sits at
  -0.19 rather than 0.
- RMSE 0.960 log units is a factor of nine in concentration. Ask whether they
  would sign a REACH dossier on that. The honest answer is that these models are
  used for screening and prioritisation, not for setting limits — which is a
  statement about the *decision*, not about the R².
- MLOGP alone gets 0.42 of the 0.56. Expect someone to propose that the other
  five descriptors are nearly useless; park that claim and return to it in
  session 4 with permutation importance.

**Common student errors.**

- Scoring on `X_train` and reporting it. Let this happen in EX 1.4 before
  correcting it.
- Reading a high LC50 as low toxicity. Flag the sign of the log at the start and
  again when interpreting residuals.
- Using `train_test_split` without `random_state` and then being unable to
  reproduce their own number — a useful accident if it occurs.

---

## Session 2 — Where the honest number comes from

**Objectives.** Quantify split-to-split variability; locate every data-dependent
step relative to the resampling loop; measure three leakage mechanisms and
discover that only one of them bites here.

**Expected numbers (fish).**

| experiment | result |
|---|---|
| EX 2.1 — 200 single 75/25 splits, RF | mean 0.612, sd 0.045, min 0.461, max 0.725; 5-95% 0.541-0.679; best is seed 69 |
| EX 2.2 — RepeatedKFold | mean 0.603 / 0.602 / 0.605 / 0.609 / 0.613 for 1, 2, 3, 5, 10 repeats; sd *rises* 0.027 → 0.051 and settles |
| EX 2.3 — scaling | kNN 0.564 → 0.609; Ridge 0.556 → 0.557; RF 0.603 → 0.603 |
| EX 2.4 — scaler fitted on all data | 0.6131 vs 0.6093, **optimism +0.004** |
| EX 2.5 — select-then-CV, n=60, p=2000 pure noise | **+0.386** vs -0.177 done correctly |
| EX 2.6 — same, other regimes, mean of 5 draws | n=60/p=500: **+0.234** (sd 0.186); n=60/p=2000: **+0.421** (sd 0.127); n=120/p=2000: **+0.251** (sd 0.089); n=300/p=2000: **+0.151** (sd 0.047). Done correctly, all four sit between -0.34 and -0.60 |
| EX 2.7 — duplicates | 908 rows, 775 distinct descriptor vectors, **217 rows with a twin** (84 groups; `X.duplicated().sum()` = 133 counts excess rows, not rows involved) |
| EX 2.7 — KFold vs GroupKFold, aggregate | k=1: 0.406/0.415; k=5: 0.609/0.606; RF(200): 0.603/0.605 |
| EX 2.7 — k=1 scored on affected rows only | **0.649 random vs 0.519 grouped**; singletons 0.375 vs 0.397 |

**Discussion prompts.**

- After EX 2.1, ask the room to write the abstract sentence they could publish
  from seed 69 alone (R² = 0.725). Then ask what would have to be true of the literature if
  this is as easy as it looks. This is the moment the course lands or does not.
- EX 2.4 is the null. The wrong lesson ("so pipelines are pedantry") will be
  offered by someone; the discussion to steer toward is *why* it is small here —
  a mean over 908 molecules barely moves when you drop 180 — and what would make
  it large. Ask them to predict the result of EX 2.5 before running it.
- EX 2.6 averages five draws per cell on purpose, and the reason is worth
  showing. An earlier version of this notebook cell drew once per row of the
  table, re-seeding the generator for each row, and produced +0.386 at n=60 and
  +0.387 at n=120 (p=2000 in both) — no visible n effect at all. The
  draw-to-draw sd at n=60, p=2000 is 0.127, so a single draw carries about the
  same uncertainty as the 0.170 gap it is supposed to reveal; averaging five
  draws brings the standard error of each cell to roughly 0.06 and the trend
  (0.421 → 0.251 → 0.151) appears. If anyone complains that five draws is fussy,
  that is the exercise working: the lesson of EX 2.1 applies to the artefact as
  much as to the model.
- EX 2.7 part 3 is the methodological heart of the session. The aggregate gap is
  0.004 and the gap on affected rows is 0.130. Both are correct. Make them say
  out loud which one they would quote if they wanted to keep a random split, and
  which a referee should ask for.
- Connect to their own work: who has technical replicates, repeated measures per
  animal, multiple cell lines per donor, or homologous sequences? In those
  designs the affected fraction is not 24% but most of the dataset.

**Common student errors.**

- In EX 2.1, re-seeding the *model* instead of the *split*, which collapses the
  spread to almost nothing. Worth catching early; it is the same confusion as
  not knowing which source of variability one is measuring.
- In EX 2.5, using `SelectKBest` fitted inside a loop over folds but on `X`
  rather than `X.iloc[train]`. The result looks correct-by-construction and is
  not.
- Expecting `GroupKFold` to improve the score. It lowers it, by design. Say
  explicitly that correct methodology usually costs performance on paper, and
  that this is why it needs defending on principle.

---

## Session 3 — Choosing a model without fooling yourself

**Objectives.** Compare models with error bars and paired folds; recognise how
much of a leaderboard is noise; measure selection bias with nested CV; read a
learning curve as a budget decision.

**Expected numbers (fish, 5-fold × 5 repeats = 25 folds).**

| model | R² | sd | RMSE |
|---|---|---|---|
| RandomForest (500) | 0.610 | 0.050 | 0.902 |
| kNN (k=5) | 0.609 | 0.057 | 0.902 |
| HistGB | 0.601 | 0.057 | 0.911 |
| Ridge (α=1) | 0.560 | 0.063 | 0.956 |
| LinearRegression | 0.560 | 0.063 | 0.956 |
| Dummy | -0.008 | 0.014 | 1.454 |

- EX 3.2 paired: RF − Ridge = **+0.049 ± 0.041** per fold; forest wins 84% of 25
  folds.
- EX 3.3 validation curve, kNN: k=1 → 0.406, k=3 → 0.584, k=5 → 0.609,
  **k=8 → 0.619**, k=12 → 0.615, k=20 → 0.608, k=30 → 0.597, k=50 → 0.569.
- EX 3.4 leaderboard, 39 configs on one shared 5-fold split: `rf(depth=None,
  max_features=0.3)` 0.631, `kNN8` 0.619, `rf(None, 0.6)` 0.617, `rf(8, 0.3)`
  0.616, `kNN12` 0.615, `rf(8, 0.6)` 0.610, `kNN5` 0.609, `kNN20` 0.608.
  **1st − 8th = 0.023**, against a single-split sd of 0.045.
- EX 3.5 nested CV at n=908: 0.645 ± 0.048 vs flat winner 0.631, i.e. optimism
  **-0.014** — not detectable. Four of the five outer folds select
  `rf(max_depth=None, max_features=0.3)`; the fifth selects the same forest with
  `max_features=0.6`.
- EX 3.6 nested CV at n=120 with `n_estimators=100`, three subsamples:
  optimism **+0.126, +0.030, +0.029**, mean **+0.062**.
- EX 3.7 learning curves: RF train 0.918→0.940, CV 0.522→0.603, still rising at
  n=726. Ridge train ≈0.58, CV 0.421→0.557, flat from about n=400.

**Discussion prompts.**

- EX 3.1 gives two defensible readings of one table: "no model → linear model"
  is a large unambiguous step, "linear → forest" is the size of the error bars.
  Ask for both in one sentence each.
- EX 3.2 part 3 is the statistics point of the session. A paired t-test on 25
  folds gives p < 10⁻⁵ and is meaningless, because `n_repeats` is a free
  parameter: the p-value can be driven arbitrarily low without a single new
  molecule. Students who have had a classical statistics course find this
  genuinely unsettling, which is the right reaction.
- EX 3.4: three different model families occupy the top eight, within 0.023.
  Re-run with a different `random_state` in `cv_shared` live, in front of the
  room, and watch the winner change. Ninety seconds, and it is the cheapest
  lesson in the course.
- EX 3.5/3.6 together: the same code gives -0.01 at n=908 and +0.06 at n=120,
  with one subsample at +0.13. Ask what that implies for a paper that reports the
  best of a 500-configuration search on a 50-patient cohort.
- EX 3.7 is the applied question: the forest's curve is still climbing, so more
  molecules help it; ridge has plateaued, so more molecules do nothing for it.
  Have them write the one-sentence recommendation to a lab head with €20k.

**Common student errors.**

- Comparing models by their means while the sds overlap, without the paired
  analysis. Ask for the win fraction every time a comparison is claimed.
- Building the leaderboard with a fresh `KFold` per configuration, which
  destroys the pairing. Insist on one shared `cv_shared` object.
- In EX 3.5, fitting the winning configuration on all the data rather than on
  the outer training fold. This silently reintroduces exactly the leak nested CV
  exists to avoid; check their loop.
- Reading nested CV as "a better model". It scores a *procedure*, and the object
  you deploy is still the one fitted on everything.

---

## Session 4 — What the model knows, and where it stops working

**Objectives.** Build an empirical null for a whole pipeline; read residuals
against a descriptor; distinguish two importance measures and what each
licenses; demonstrate that a cross-validated score does not transfer outside the
training domain; specify an applicability domain.

**Expected numbers.**

| experiment | result |
|---|---|
| EX 4.1 — y-scrambling, RF, 30 shuffles | real **+0.603**; null **-0.187 ± 0.040**, max -0.086 |
| EX 4.2 — out-of-fold RF | RMSE **0.910**; RMSE by MLOGP tertile **1.031 (low) / 0.724 (mid) / 0.946 (high)** — worst on the *least* lipophilic third |
| EX 4.2 — five largest residuals | all but one are under-predictions of highly toxic, low-to-moderate-MLOGP compounds (LC50 8.13 predicted 3.15; 9.61 → 5.54; 6.42 → 2.65; 7.81 → 4.10) |
| EX 4.2 — identical descriptor vectors | 84 groups, 217 rows; mean within-group LC50 sd **0.320**, largest range **3.086** log units |
| EX 4.3 — fish importance (impurity / permutation) | MLOGP 0.562/0.410, SM1_Dz(Z) 0.171/0.235, GATS1i 0.114/0.097, CIC0 0.093/0.031, NdsCH 0.033/**0.088**, NdssC 0.026/0.012 |
| EX 4.4 — aquatic, r(TPSA, SAacc) = 0.860 | TPSA 0.108/**0.217**, SAacc 0.090/0.067, MLOGP 0.366/0.284, RDCHI 0.187/0.103, GATS1p 0.174/**0.039**, nN 0.027/**-0.005** |
| EX 4.5 — fish, top MLOGP quartile held out | **R² 0.037**, RMSE 1.195 vs random split 0.607; mean LC50 3.62 → 5.41 |
| EX 4.5 — aquatic, same | **R² -0.657**, RMSE 1.649 vs random split 0.489; mean LC50 4.23 → 5.94 |
| EX 4.5 — fish, top CIC0 quartile instead | R² 0.282 — "out of domain" is not one thing |
| EX 4.6 — error by distance quartile | RMSE **0.512 / 0.772 / 0.901 / 1.428** from nearest to farthest; mean distance 0.79 for a random held-out sample vs **1.61** for the high-MLOGP molecules |

**Discussion prompts.**

- The y-scrambling null is centred at -0.19, not 0. Ask why a flexible model
  fitted to pure noise is *worse* than the mean on held-out data. Then ask what
  the null would look like if the pipeline contained the session-2 selection
  leak — the answer is that it moves up with the real score, which is why the
  null has to be computed for the *whole* pipeline, re-selecting inside each
  shuffle.
- EX 4.2's residual pattern is the one place where the chemistry is worth a
  digression. Error is *highest* in the low-MLOGP tertile, and the largest
  residuals are under-predictions of compounds that are very toxic without being
  lipophilic. That is the signature of **excess toxicity**: narcosis sets a
  baseline as a function of log P, and compounds acting through a specific
  target sit far above it. A model dominated by `MLOGP` reproduces the baseline
  and must miss them, and no amount of model tuning fixes it — the information
  is not in the six descriptors. Note the direction of failure: the model
  under-predicts toxicity for the most toxic compounds, which is the unsafe
  direction for a screening tool. (The dataset carries no structures, so treat
  the mechanism as the standard explanation for this residual pattern, not as
  an identification of those five molecules.)
- EX 4.2's within-group spread (0.320 log units mean, 3.086 worst) is a floor on
  achievable RMSE for *any* model on these six descriptors. Current RMSE is
  0.910, so the representation is not yet the binding constraint. This is the
  argument to have ready when a student proposes a neural network: compute the
  ceiling before buying capacity.
- EX 4.3: `NdsCH` has impurity importance 0.033 and permutation importance
  0.088. Impurity importance is computed on training data and favours features
  with many distinct values (`MLOGP` has 559 unique values; `NdsCH` has 5), so a
  count feature is systematically understated. Ask which of the two they would
  put in a figure captioned "descriptors driving toxicity" — and then whether
  either is adequate for that caption.
- EX 4.4 is the limit of importance analysis. Dropping `SAacc`, correlated at
  0.86 with `TPSA(Tot)`, moves CV R² only from 0.512 to 0.501 — inside the ±0.08
  fold spread — while the importances rearrange: `H-050` rises from 0.029 to
  0.089 and `TPSA(Tot)` *falls* from 0.217 to 0.190. Worth drawing out that the
  redistribution does not go to the dropped descriptor's closest partner, which
  is hard to reconcile with reading importance as mechanism. `nN` also carries a
  slightly negative permutation importance. Insist on the phrasing: importance
  is a property of *this fitted model on this descriptor set*, not of the
  chemistry.
- EX 4.5 is the headline result of the course: 0.61 → 0.04 on fish, 0.49 →
  -0.66 on aquatic, from a split that any medicinal chemist would call routine.
  Draw out the direction of the error — the model under-predicts toxicity for
  the most toxic compounds, because a forest cannot extrapolate beyond the
  target range in its leaves. Ask which way a regulator would want it to fail.

**Common student errors.**

- Running the y-scrambling shuffle *inside* a pipeline that re-selects features,
  but shuffling only once rather than per repeat. The null then has no spread.
- Interpreting a near-zero permutation importance as biological irrelevance.
- In EX 4.5, scaling or imputing using the full dataset before making the
  extrapolation split, which quietly leaks the held-out chemistry. Same mistake
  as session 2, now in a context where it would matter.
- Concluding "the model is broken". It is not: it is being used outside its
  domain, which is a statement about deployment, not about fitting.

---

## Capstone marking

One notebook plus a methods paragraph of ≤250 words, on the aquatic dataset.
Suggested weighting out of 100.

| Component | Marks |
|---|---|
| Baseline reported, in R² and as a fold-error | 10 |
| Model selection: ≥3 families × ≥3 settings, number of configurations stated, nested or explicitly labelled an upper bound | 20 |
| Repeated CV with a spread; runner-up comparison | 15 |
| Leakage audit: every data-dependent step located relative to the loop; duplicate count reported and acted on | 20 |
| y-scrambling null for the final pipeline | 10 |
| Applicability domain: one extrapolation experiment, one indicator, one explicit "do not use for" statement | 20 |
| One honest negative result, with the number | 5 |

**Reference figures for the aquatic dataset** (5-fold × 5 repeats): dummy
-0.008 ± 0.011 (RMSE 1.663, 46×), ridge(α=1) 0.458 ± 0.058 (RMSE 1.216, 16×),
kNN(k=5) 0.487 ± 0.082, HistGB 0.509 ± 0.094, RF(500) 0.533 ± 0.083 (RMSE
1.125, 13×). Extrapolation on the top MLOGP quartile: -0.657.

A submission landing near 0.53 ± 0.08 with a complete audit is a strong pass.
**Mark the audit above the performance** and say so when briefing: anything
materially above ~0.55 on this dataset is more likely to indicate a leak than a
better model, and finding that leak is worth more marks than the score it
inflated. A submission that reports 0.50 with a correct audit should outrank one
reporting 0.60 without.

Legitimate answers to requirement 7 (the honest negative) include: the scaler
leak measured at +0.004; grouped versus random splitting, where only 16 of the
546 aquatic rows share a descriptor vector with another row (against 217 of 908
in the fish data), so the measured gap is +0.000 for a random forest and +0.016
for kNN(k=1); dropping a collinear descriptor with no performance change;
or selection bias at this n. Any of these earns full marks *provided the number
is reported*.

---

## Rebuilding the notebooks

`build_notebooks.py` (kept outside the student package) generates all eight
notebooks from a single source, so the student scaffold and the solution cannot
drift. Each exercise is one `code(student, solution)` call. Edit there, re-run,
and re-execute the solutions headless to confirm they still work:

```bash
uv run python build_notebooks.py          # regenerates all eight notebooks
uv run python run_solutions.py            # executes all four solution notebooks
```

`run_solutions.py` aborts on the first cell that raises, so it works as a
regression test. Executed copies with all outputs land in `executed/`.

## Attribution

Data: UCI Machine Learning Repository, QSAR fish toxicity and QSAR aquatic
toxicity, contributed by the Milano Chemometrics and QSAR Research Group
(M. Cassotti, D. Ballabio, R. Todeschini, V. Consonni). The underlying LC50
values derive from the ECOTOX aquatic toxicology database.
