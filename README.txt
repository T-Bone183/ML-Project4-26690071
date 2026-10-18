ML 441/741 Assignment 4 -- Ensemble Learning for Network Attack Classification
Student number: 26690071

OVERVIEW
--------
This project investigates whether the classification tree from Assignment 2, trained on
the modified UNSW-NB15 intrusion-detection dataset, can be improved -- either by changing
the underlying tree algorithm (CART vs C4.5) or by aggregating trees into an ensemble
(single tree vs random forest). Four models are compared: CART and C4.5, each as a single
tree and as a bagged random forest. The CART models use scikit-learn; the C4.5 models (both
the single tree and the forest) are implemented from scratch in Java.

REPOSITORY STRUCTURE
--------------------
Report/   LaTeX source and compiled PDF of the report.
Code/     Python notebook and the hand-written Java C4.5 implementation.

REPORT/
-------
  26690071RW441assignment4.pdf   compiled report (the submission)
  26690071RW441assignment4.tex   LaTeX source
  references.bib                 bibliography
  confusion_matrices.pdf         figure used in the report
  IEEEtran.cls, IEEEtran.bst     IEEE conference template files
  Makefile                       builds the PDF

  To rebuild the report (requires a TeX distribution with latexmk):
      cd Report
      make

CODE/
-----
  q1.ipynb        Jupyter notebook that drives the whole experiment: data loading and
                  preprocessing, the CART models (scikit-learn), exporting the data splits
                  for the Java side, invoking the Java C4.5 program, and scoring every model
                  with a single macro-F1 implementation.
  c45-java/       the hand-written C4.5 tree and random forest.
      C45.java           the implementation (classes Dataset, Node, Split, C45Tree,
                         RandomForest, Metrics, CSV, and the C45 driver)
      README.md          describes the Java pipeline and how it is invoked
      best_standard.txt  tuned hyperparameters selected for the C4.5 single tree
      best_forest.txt    tuned hyperparameters selected for the C4.5 forest

  How it fits together:
    1. The notebook loads networkTraffic.csv, preprocesses it, and tunes and evaluates the
       CART models with scikit-learn.
    2. For the C4.5 models it exports the tuning and training/test splits to
       c45-java/tune.csv and c45-java/train_test.csv.
    3. It compiles the Java (javac C45.java) and runs it, which tunes
       (java C45 Tune Standard|Forest) and evaluates (java C45 Train_Test Standard|Forest),
       writing pooled out-of-fold predictions to CSV.
    4. The notebook reads those predictions back and scores all four models with the same
       scikit-learn macro-F1, so the results are directly comparable.

  Requirements:
    - Python 3 with pandas, numpy, scikit-learn and matplotlib
    - A Java Development Kit (JDK 17 or later)

NOTE ON DATA
------------
The dataset (networkTraffic.csv, ~47 MB) and the large generated files (tune.csv,
train_test.csv and predictions_*.csv) are not committed, to keep the repository small. To
run the notebook end to end, place networkTraffic.csv in the Code/ directory; the split and
prediction files are then regenerated automatically.
