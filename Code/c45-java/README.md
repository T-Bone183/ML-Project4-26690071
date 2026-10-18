# C4.5 (Java) — Assignment 4

Custom C4.5 decision tree + bagging/forest, implemented in Java for speed. The notebook
(`../q1.ipynb`) handles the shared data split, the CART baselines, and **all reported metrics**;
this Java side trains the C4.5 models and writes back predictions, so every model in the 2×2 is
scored by the identical scikit-learn code.

## Build & run (no dependencies — JDK only)

```bash
cd c45-java
javac C45.java
java C45 <command> <args...>
```

## Pipeline / CSV contract

```
q1.ipynb  --export-->  tune.csv, train_test.csv, feature_types.json
                       (C4.5-native: raw features, no one-hot; same rows as the CART split, by index)
                               |
                               v
   Java (this project)  -->  grid-search (OOB for forest) -> best hyperparameters
                       -->  stratified k-fold on train_test -> out-of-fold predictions
                               |
                               v
q1.ipynb  <--import--  predictions_c45_tree.csv, predictions_c45_forest.csv
                       (row-aligned, with a `fold` column)
                               |
                               v
   q1.ipynb computes macro-F1 / classification_report / confusion_matrix (sklearn)
```

- **Java computes a macro-F1 only for its own grid-search *selection*** (does not need to match sklearn).
- **Final reported numbers are computed in the notebook** from the exported predictions.

## Structure — single file `C45.java` (top-level classes)

| Class | Responsibility |
|---|---|
| `C45`          | Entry point / orchestration: read CSV, tune, k-fold, write predictions. |
| `Dataset`      | Load C4.5-native CSV; columnar primitive storage; int-coded categoricals. |
| `Node`         | One tree node (internal split or leaf class distribution). |
| `Split`        | The chosen split at a node (one per node, not per candidate). |
| `C45Tree`      | Entropy / info gain / gain ratio; multi-way + numeric splits; REP pruning (flag). |
| `RandomForest` | Bagging + subspace sampling over `C45Tree`; out-of-bag error. |
| `Metrics`      | Macro-F1 for Java-side tuning selection only. |
| `Csv`          | Minimal CSV read / write. |

No missing-value handling is needed: `service`-missing is encoded as its own category upstream,
so the C4.5-native data is complete (no fractional-instance method). Classification only.

## Validation target
Reproduce the Topic 4 film-success worked example (slides 20–22): H(D)=1, Gain(Country)=0.2756,
Gain(BigStar)=0.0349, Gain(Genre)=0.17 → Country split first. Lock this down before trusting any results.
