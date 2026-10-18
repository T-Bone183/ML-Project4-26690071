import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.util.concurrent.*;
import java.math.*;;


/**
 * C4.5 (classification only, using REP) decision tree with bagging/forest

 */
public class C45 {

    //fixed experiment setup (matches the notebook export)
    static final List<String> CATEGORICAL =
        Arrays.asList("proto", "service", "state", "ct_state_ttl");
    static final String LABEL = "attack_cat";
    static final int K_FOLDS = 5;
    static final long SEED = 42L;
    //REP uses this fraction of each training fold to grow, the rest to prune
    static final double GROW_FRACTION = 0.7;

    /**
     * args[0] = Tune | Train_Test   (Tune reads tune.csv, Train_Test reads train_test.csv)
     * args[1] = Standard | Forest   (single pruned C4.5 tree, or the bagged forest)
     * Tune grid-searches and writes best_{standard,forest}.txt; Train_Test reads that file,
     * runs stratified 5-fold and writes predictions_{standard,forest}.csv for the notebook to score.
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("usage: java C45 <Tune|Train_Test> <Standard|Forest>");
            return;
        }
        String mode = args[0];
        String model = args[1];

        //pick the input split from the mode
        String inputFile;
        if (mode.equals("Tune")) {
            inputFile = "tune.csv";
        } else if (mode.equals("Train_Test")) {
            inputFile = "train_test.csv";
        } else {
            System.out.println("first argument must be Tune or Train_Test");
            return;
        }

        if (!model.equals("Standard") && !model.equals("Forest")) {
            System.out.println("second argument must be Standard or Forest");
            return;
        }

        Dataset d = Dataset.load(inputFile, CATEGORICAL, LABEL);
        //one worker per core, all parallelism lives here in main (fit stays single-threaded)
        int cores = Runtime.getRuntime().availableProcessors();
        System.out.println("loaded " + inputFile + ": " + d.nRows + " rows, "
            + d.nFeatures + " features, " + d.nClasses + " classes; using " + cores + " workers");

        ExecutorService pool = Executors.newFixedThreadPool(cores);
        try {
            if (mode.equals("Tune") && model.equals("Standard")) {
                tuneStandard(d, pool);
            } else if (mode.equals("Tune") && model.equals("Forest")) {
                tuneForest(d, pool);
            } else if (mode.equals("Train_Test") && model.equals("Standard")) {
                evalStandard(d, pool);
            } else {
                evalForest(d, pool);
            }
        } finally {
            pool.shutdown();
        }
    }

    /** Grid search the single pruned tree by stratified k-fold macro-F1; writes best_standard.txt. */
    static void tuneStandard(final Dataset d, ExecutorService pool) throws Exception {
        //grid: split criterion x minSamplesLeaf x classWeight x prune
        C45Tree.Criterion[] criteria = {C45Tree.Criterion.INFO_GAIN, C45Tree.Criterion.GAIN_RATIO};
        int[] leafSizes = {1, 5, 15, 30};
        String[] classWeights = {"none", "balanced"};
        boolean[] prunes = {true, false};

        //flatten the cross product into parallel lists of configs
        List<C45Tree.Criterion> cfgCriterion = new ArrayList<>();
        List<Integer> cfgLeaf = new ArrayList<>();
        List<String> cfgWeight = new ArrayList<>();
        List<Boolean> cfgPrune = new ArrayList<>();
        for (int a = 0; a < criteria.length; a++) {
            for (int b = 0; b < leafSizes.length; b++) {
                for (int e = 0; e < classWeights.length; e++) {
                    for (int g = 0; g < prunes.length; g++) {
                        cfgCriterion.add(criteria[a]);
                        cfgLeaf.add(leafSizes[b]);
                        cfgWeight.add(classWeights[e]);
                        cfgPrune.add(prunes[g]);
                    }
                }
            }
        }

        final int[] folds = stratifiedFolds(d, K_FOLDS, SEED);

        //one task per config, each runs a whole k-fold CV single-threaded
        List<Future<Double>> futures = new ArrayList<>();
        for (int c = 0; c < cfgCriterion.size(); c++) {
            final C45Tree.Criterion crit = cfgCriterion.get(c);
            final int minSamplesLeaf = cfgLeaf.get(c);
            final String classWeight = cfgWeight.get(c);
            final boolean prune = cfgPrune.get(c);
            futures.add(pool.submit(new Callable<Double>() {
                public Double call() {
                    return treeKFoldMacroF1(d, folds, crit, minSamplesLeaf, classWeight, prune);
                }
            }));
        }

        //collect results and keep the best
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < futures.size(); c++) {
            double score = futures.get(c).get();
            System.out.println("  criterion=" + cfgCriterion.get(c)
                + " minSamplesLeaf=" + cfgLeaf.get(c) + " classWeight=" + cfgWeight.get(c)
                + " prune=" + cfgPrune.get(c) + " -> macroF1=" + String.format("%.4f", score));
            if (score > bestScore) {
                bestScore = score;
                best = c;
            }
        }

        System.out.println("best standard: criterion=" + cfgCriterion.get(best)
            + " minSamplesLeaf=" + cfgLeaf.get(best) + " classWeight=" + cfgWeight.get(best)
            + " prune=" + cfgPrune.get(best) + " macroF1=" + String.format("%.4f", bestScore));

        //write the winning config for Train_Test to read back
        StringBuilder sb = new StringBuilder();
        sb.append("criterion=").append(cfgCriterion.get(best)).append("\n");
        sb.append("minSamplesLeaf=").append(cfgLeaf.get(best)).append("\n");
        sb.append("classWeight=").append(cfgWeight.get(best)).append("\n");
        sb.append("prune=").append(cfgPrune.get(best)).append("\n");
        writeText("best_standard.txt", sb.toString());
    }

    /**k-fold CV marco-F1 for one tree config; pools the out-of-fold predictions. */
    static double treeKFoldMacroF1(Dataset d, int[] folds, C45Tree.Criterion crit, int minSamplesLeaf, String classWeight, boolean prune) {
        int[] oofPred = new int[d.nRows];

        //apply balanced class weights once if requested (per-class constant, so one view serves every fold)
        Dataset work = d;
        if (classWeight.equals("balanced")) {
            work = reweighted(d);
            applyBalancedWeights(work);
        }

        for (int f = 0; f < K_FOLDS; f++) {
            int[][] parts = splitTrainTest(d, folds, f);
            int[] trainRows = parts[0];
            int[] testRows = parts[1];

            C45Tree tree = new C45Tree();
            tree.criterion = crit;
            tree.minSamplesLeaf = minSamplesLeaf;
            //we drive REP by hand from the held-out prune rows, so leave the flag off
            tree.prune = false;
            tree.rng = new Random(SEED + f);
            if (prune) {
                //carve a held-out prune set out of the training rows, then grow + REP
                int[][] gp = stratifiedGrowPrune(d, trainRows, GROW_FRACTION, new Random(SEED + f));
                tree.fit(work, gp[0]);
                tree.repPrune(work, gp[1]);
            } else {
                //no pruning: grow on all of the training rows
                tree.fit(work, trainRows);
            }

            for (int k = 0; k < testRows.length; k++) {
                oofPred[testRows[k]] = tree.predict(work, testRows[k]);
            }
        }
        return Metrics.macroF1(d.y, oofPred, d.nClasses);
    }

    /** Grid search the forest by out-of-bag macro-F1 (one fit per config); writes best_forest.txt. */
    static void tuneForest(final Dataset d, ExecutorService pool) throws Exception {
        C45Tree.Criterion[] criteria = {C45Tree.Criterion.INFO_GAIN, C45Tree.Criterion.GAIN_RATIO};
        //subspace sizes scaled to this dataset's feature count (sqrt, ~0.3, ~0.5)
        int[] featureCounts = {
            Math.max(1, (int) Math.round(Math.sqrt(d.nFeatures))),
            Math.max(1, (int) Math.round(0.3 * d.nFeatures)),
            Math.max(1, (int) Math.round(0.5 * d.nFeatures))
        };
        int[] leafSizes = {1, 5, 15, 30};
        String[] classWeights = {"none", "balanced"};
        int[] treeCounts = {100, 300};

        List<C45Tree.Criterion> cfgCriterion = new ArrayList<>();
        List<Integer> cfgFeatures = new ArrayList<>();
        List<Integer> cfgLeaf = new ArrayList<>();
        List<String> cfgWeight = new ArrayList<>();
        List<Integer> cfgTrees = new ArrayList<>();
        for (int a = 0; a < criteria.length; a++) {
            for (int b = 0; b < featureCounts.length; b++) {
                for (int e = 0; e < leafSizes.length; e++) {
                    for (int g = 0; g < classWeights.length; g++) {
                        for (int h = 0; h < treeCounts.length; h++) {
                            cfgCriterion.add(criteria[a]);
                            cfgFeatures.add(featureCounts[b]);
                            cfgLeaf.add(leafSizes[e]);
                            cfgWeight.add(classWeights[g]);
                            cfgTrees.add(treeCounts[h]);
                        }
                    }
                }
            }
        }

        //one task per config: fit a forest and read its OOB macro-F1
        List<Future<Double>> futures = new ArrayList<>();
        for (int c = 0; c < cfgCriterion.size(); c++) {
            final C45Tree.Criterion crit = cfgCriterion.get(c);
            final int maxFeatures = cfgFeatures.get(c);
            final int minSamplesLeaf = cfgLeaf.get(c);
            final String classWeight = cfgWeight.get(c);
            final int nTrees = cfgTrees.get(c);
            futures.add(pool.submit(new Callable<Double>() {
                public Double call() {
                    //each config gets its own view so balanced weights do not clash across threads
                    Dataset work = d;
                    if (classWeight.equals("balanced")) {
                        work = reweighted(d);
                        applyBalancedWeights(work);
                    }
                    RandomForest forest = new RandomForest();
                    forest.nTrees = nTrees;
                    forest.maxFeatures = maxFeatures;
                    forest.minSamplesLeaf = minSamplesLeaf;
                    forest.criterion = crit;
                    forest.rng = new Random(SEED);
                    forest.fit(work);
                    return forest.oobMacroF1(work);
                }
            }));
        }

        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < futures.size(); c++) {
            double score = futures.get(c).get();
            System.out.println("  criterion=" + cfgCriterion.get(c)
                + " maxFeatures=" + cfgFeatures.get(c) + " minSamplesLeaf=" + cfgLeaf.get(c)
                + " classWeight=" + cfgWeight.get(c) + " nTrees=" + cfgTrees.get(c)
                + " -> oobMacroF1=" + String.format("%.4f", score));
            if (score > bestScore) {
                bestScore = score;
                best = c;
            }
        }

        System.out.println("best forest: criterion=" + cfgCriterion.get(best)
            + " maxFeatures=" + cfgFeatures.get(best) + " minSamplesLeaf=" + cfgLeaf.get(best)
            + " classWeight=" + cfgWeight.get(best)
            + " nTrees=" + cfgTrees.get(best) + " oobMacroF1=" + String.format("%.4f", bestScore));

        StringBuilder sb = new StringBuilder();
        sb.append("criterion=").append(cfgCriterion.get(best)).append("\n");
        sb.append("maxFeatures=").append(cfgFeatures.get(best)).append("\n");
        sb.append("minSamplesLeaf=").append(cfgLeaf.get(best)).append("\n");
        sb.append("classWeight=").append(cfgWeight.get(best)).append("\n");
        sb.append("nTrees=").append(cfgTrees.get(best)).append("\n");
        writeText("best_forest.txt", sb.toString());
    }

    /** Train_Test: read best_standard.txt, run stratified k-fold, write out-of-fold predictions. */
    static void evalStandard(final Dataset d, ExecutorService pool) throws Exception {
        Map<String, String> best = readBestParams("best_standard.txt");
        final C45Tree.Criterion crit = C45Tree.Criterion.valueOf(best.get("criterion"));
        final int minSamplesLeaf = Integer.parseInt(best.get("minSamplesLeaf"));
        String classWeight = best.get("classWeight");
        final boolean prune = Boolean.parseBoolean(best.get("prune"));

        //balanced weights only depend on the class, so one shared view is safe across the fold tasks
        Dataset tmp = d;
        if (classWeight.equals("balanced")) {
            tmp = reweighted(d);
            applyBalancedWeights(tmp);
        }
        final Dataset work = tmp;

        final int[] folds = stratifiedFolds(d, K_FOLDS, SEED);
        //each fold writes only its own test rows, so this shared array is filled without clashes
        final int[] preds = new int[d.nRows];

        //one task per fold
        List<Future<Object>> futures = new ArrayList<>();
        for (int f = 0; f < K_FOLDS; f++) {
            final int foldIndex = f;
            futures.add(pool.submit(new Callable<Object>() {
                public Object call() {
                    int[][] parts = splitTrainTest(d, folds, foldIndex);
                    int[] trainRows = parts[0];
                    int[] testRows = parts[1];

                    C45Tree tree = new C45Tree();
                    tree.criterion = crit;
                    tree.minSamplesLeaf = minSamplesLeaf;
                    tree.prune = false;
                    tree.rng = new Random(SEED + foldIndex);
                    if (prune) {
                        int[][] gp = stratifiedGrowPrune(d, trainRows, GROW_FRACTION, new Random(SEED + foldIndex));
                        tree.fit(work, gp[0]);
                        tree.repPrune(work, gp[1]);
                    } else {
                        tree.fit(work, trainRows);
                    }

                    for (int k = 0; k < testRows.length; k++) {
                        preds[testRows[k]] = tree.predict(work, testRows[k]);
                    }
                    return null;
                }
            }));
        }
        for (int f = 0; f < futures.size(); f++) {
            futures.get(f).get();
        }

        writePredictionsFile("predictions_standard.csv", d, folds, preds);
        System.out.println("wrote predictions_standard.csv (" + d.nRows + " rows)");
    }

    /** Train_Test: read best_forest.txt, run stratified k-fold, write out-of-fold predictions. */
    static void evalForest(final Dataset d, ExecutorService pool) throws Exception {
        Map<String, String> best = readBestParams("best_forest.txt");
        final C45Tree.Criterion crit = C45Tree.Criterion.valueOf(best.get("criterion"));
        final int maxFeatures = Integer.parseInt(best.get("maxFeatures"));
        final int minSamplesLeaf = Integer.parseInt(best.get("minSamplesLeaf"));
        final String classWeight = best.get("classWeight");
        final int nTrees = Integer.parseInt(best.get("nTrees"));

        final int[] folds = stratifiedFolds(d, K_FOLDS, SEED);
        final int[] preds = new int[d.nRows];

        List<Future<Object>> futures = new ArrayList<>();
        for (int f = 0; f < K_FOLDS; f++) {
            final int foldIndex = f;
            futures.add(pool.submit(new Callable<Object>() {
                public Object call() {
                    int[][] parts = splitTrainTest(d, folds, foldIndex);
                    int[] trainRows = parts[0];
                    int[] testRows = parts[1];

                    //the forest trains on whatever Dataset it is handed, so give it a train-only view
                    Dataset trainView = subset(d, trainRows);
                    if (classWeight.equals("balanced")) {
                        //balance against the class frequencies in this fold's training rows
                        applyBalancedWeights(trainView);
                    }
                    RandomForest forest = new RandomForest();
                    forest.nTrees = nTrees;
                    forest.maxFeatures = maxFeatures;
                    forest.minSamplesLeaf = minSamplesLeaf;
                    forest.criterion = crit;
                    forest.rng = new Random(SEED + foldIndex);
                    forest.fit(trainView);

                    //codes and feature indices match the original, so predict the test rows straight off d
                    for (int k = 0; k < testRows.length; k++) {
                        preds[testRows[k]] = forest.predict(d, testRows[k]);
                    }
                    return null;
                }
            }));
        }
        for (int f = 0; f < futures.size(); f++) {
            futures.get(f).get();
        }

        writePredictionsFile("predictions_forest.csv", d, folds, preds);
        System.out.println("wrote predictions_forest.csv (" + d.nRows + " rows)");
    }

    /** Split all rows into {trainRows, testRows} for the given fold. */
    static int[][] splitTrainTest(Dataset d, int[] folds, int foldIndex) {
        int testCount = 0;
        for (int i = 0; i < d.nRows; i++) {
            if (folds[i] == foldIndex) {
                testCount++;
            }
        }
        int[] trainRows = new int[d.nRows - testCount];
        int[] testRows = new int[testCount];
        int tr = 0;
        int te = 0;
        for (int i = 0; i < d.nRows; i++) {
            if (folds[i] == foldIndex) {
                testRows[te] = i;
                te++;
            } else {
                trainRows[tr] = i;
                tr++;
            }
        }
        return new int[][] {trainRows, testRows};
    }

    /** Stratified k-fold assignment: returns the fold index (0..k-1) for every row. */
    static int[] stratifiedFolds(Dataset d, int k, long seed) {
        int[] fold = new int[d.nRows];
        Random rng = new Random(seed);

        //group row indices by class
        List<List<Integer>> byClass = new ArrayList<>();
        for (int c = 0; c < d.nClasses; c++) {
            byClass.add(new ArrayList<>());
        }
        for (int i = 0; i < d.nRows; i++) {
            byClass.get(d.y[i]).add(i);
        }

        //shuffle each class then deal its rows round-robin across the folds
        for (int c = 0; c < d.nClasses; c++) {
            List<Integer> group = byClass.get(c);
            for (int i = group.size() - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = group.get(i);
                group.set(i, group.get(j));
                group.set(j, tmp);
            }
            for (int i = 0; i < group.size(); i++) {
                fold[group.get(i)] = i % k;
            }
        }
        return fold;
    }

    /** Stratified split of rows into {grow, prune}; growFraction of each class goes to grow. */
    static int[][] stratifiedGrowPrune(Dataset d, int[] rows, double growFraction, Random rng) {
        List<List<Integer>> byClass = new ArrayList<>();
        for (int c = 0; c < d.nClasses; c++) {
            byClass.add(new ArrayList<>());
        }
        for (int k = 0; k < rows.length; k++) {
            int idx = rows[k];
            byClass.get(d.y[idx]).add(idx);
        }

        List<Integer> grow = new ArrayList<>();
        List<Integer> prune = new ArrayList<>();
        for (int c = 0; c < d.nClasses; c++) {
            List<Integer> group = byClass.get(c);
            for (int i = group.size() - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = group.get(i);
                group.set(i, group.get(j));
                group.set(j, tmp);
            }
            int nGrow = (int) Math.round(growFraction * group.size());
            for (int i = 0; i < group.size(); i++) {
                if (i < nGrow) {
                    grow.add(group.get(i));
                } else {
                    prune.add(group.get(i));
                }
            }
        }

        int[] growArr = new int[grow.size()];
        for (int i = 0; i < grow.size(); i++) {
            growArr[i] = grow.get(i);
        }
        int[] pruneArr = new int[prune.size()];
        for (int i = 0; i < prune.size(); i++) {
            pruneArr[i] = prune.get(i);
        }
        return new int[][] {growArr, pruneArr};
    }

    /** Build a Dataset view holding only the given rows (same coding, full class set kept). */
    static Dataset subset(Dataset d, int[] rows) {
        Dataset s = new Dataset();
        s.nRows = rows.length;
        s.nFeatures = d.nFeatures;
        //keep the full class set even if a fold's training rows miss a rare class
        s.nClasses = d.nClasses;
        s.featureNames = d.featureNames;
        s.isCategorical = d.isCategorical;
        //reuse the original code -> label maps so codes stay consistent with d
        s.categories = d.categories;
        s.cols = new double[d.nFeatures][rows.length];
        s.y = new int[rows.length];
        s.weight = new double[rows.length];
        for (int k = 0; k < rows.length; k++) {
            int idx = rows[k];
            for (int ff = 0; ff < d.nFeatures; ff++) {
                s.cols[ff][k] = d.cols[ff][idx];
            }
            s.y[k] = d.y[idx];
            s.weight[k] = 1.0;
        }
        return s;
    }

    /** Overwrite a dataset's row weights in place with 'balanced' weights: weight = N / (nClasses * count_c). */
    static void applyBalancedWeights(Dataset d) {
        //count how many rows fall in each class
        int[] classCount = new int[d.nClasses];
        for (int i = 0; i < d.nRows; i++) {
            classCount[d.y[i]]++;
        }
        //rare classes get a larger weight, common classes a smaller one (averages out to 1)
        for (int i = 0; i < d.nRows; i++) {
            int c = d.y[i];
            d.weight[i] = (double) d.nRows / (d.nClasses * classCount[c]);
        }
    }

    /** A dataset view sharing everything with d (same coding) but with its own weight array. */
    static Dataset reweighted(Dataset d) {
        Dataset s = new Dataset();
        s.nRows = d.nRows;
        s.nFeatures = d.nFeatures;
        s.nClasses = d.nClasses;
        s.featureNames = d.featureNames;
        s.isCategorical = d.isCategorical;
        s.categories = d.categories;
        //share the (read-only during fitting) feature columns and labels
        s.cols = d.cols;
        s.y = d.y;
        //fresh weight array so parallel configs do not clash
        s.weight = new double[d.nRows];
        for (int i = 0; i < d.nRows; i++) {
            s.weight[i] = 1.0;
        }
        return s;
    }

    /** Read a key=value best-params file (one entry per line). */
    static Map<String, String> readBestParams(String path) {
        if (!Files.exists(Paths.get(path))) {
            throw new IllegalStateException("missing " + path + " - run the matching Tune step first");
        }
        Map<String, String> params = new HashMap<>();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(path))) {
            String line = br.readLine();
            while (line != null) {
                if (!line.isEmpty()) {
                    int eq = line.indexOf('=');
                    String key = line.substring(0, eq);
                    String value = line.substring(eq + 1);
                    params.put(key, value);
                }
                line = br.readLine();
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Couldn't read " + path, e);
        }
        return params;
    }

    /** Write a small text file (used for the best-params files). */
    static void writeText(String path, String contents) {
        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(path))) {
            bw.write(contents);
        } catch (IOException e) {
            throw new UncheckedIOException("Couldn't write " + path, e);
        }
    }

    /** Assemble row_id/fold/prediction columns and hand off to CSV.writePredictions. */
    static void writePredictionsFile(String path, Dataset d, int[] folds, int[] preds) {
        int[] rowIds = new int[d.nRows];
        for (int i = 0; i < d.nRows; i++) {
            rowIds[i] = i;
        }
        CSV.writePredictions(path, rowIds, folds, preds);
    }
}

/**
 * Column order dataset: primitive storage without an object-per-row for performance
 * Categorical features are integer-coded; numeric kept as doubles.
 */
class Dataset {
    // --- design (columnar) ---
    int nRows;
    int nFeatures;
    int nClasses;                 //# classes M
    String[] featureNames;
    boolean[] isCategorical;      //true = categorical, false = numeric
    double[][] cols;              //Data in column order, cols[f][i] -> numeric value, or cat. code (as double)
    String[][] categories;        //categories[f] : code -> original label (null if numeric)
    int[] y;                      //target class per row (0 to M-1)
    double[] weight;              //per-row weight for bagging set to 1 by default

    /** Load a C4.5-native CSV (raw features + label); `categorical` names the categorical columns. */
    static Dataset load(String csvPath, List<String> categorical, String labelColumn) {
        List<String[]> rawImport = CSV.read(csvPath);
        //first row is the names of the features
        String[] features = rawImport.get(0);
        //nFeatures = featureNames.length;

        Dataset data = new Dataset();
        //subtract one for the feature names
        data.nRows = rawImport.size() - 1;
        //subtract one for the target feature
        data.nFeatures = features.length - 1;

        //locate the target column by name inside 'features'
        int targetIdx = -1;
        for (int c = 0; c < features.length; c++)
            if (features[c].equals(labelColumn)) { 
                targetIdx = c; 
                break; 
            }
        if (targetIdx < 0) {
            throw new IllegalArgumentException("Target column not found: " + labelColumn);
        }

        //map each feature index f -> raw CSV column index (every column except the target)
        //Parallel data structures: featCol <-> featureNames
        int[] featCol = new int[data.nFeatures];
        data.featureNames = new String[data.nFeatures];
        for (int c = 0, f = 0; c < features.length; c++) {
            if (c == targetIdx) continue;
            featCol[f] = c;
            data.featureNames[f] = features[c];
            f++;
        }

        //feature types  and column-mayor layout
        data.isCategorical = new boolean[data.nFeatures];
        data.cols = new double[data.nFeatures][data.nRows];
        data.categories = new String[data.nFeatures][];
        data.y = new int[data.nRows];
        data.weight = new double[data.nRows];

        //For each index corresponding to a categorical feature, stores a list of all detinct categories
        //If index corresponds to a numeric feature, the list is null
        List<List<String>> catValues = new ArrayList<>();
        //iterate over all of the features, checking which are categorical
        for (int f = 0; f < data.nFeatures; f++) {
            //'categorical' -> list of all categorical features in the dataset
            data.isCategorical[f] = categorical.contains(data.featureNames[f]);
            if (data.isCategorical[f]) {
                //categorical feat -> add list to store its
                catValues.add(new ArrayList<>());   
            } else {
                //numeric feat -> null
                catValues.add(null);                
            }
        }

        // fill the columns row by row
        int maxClass = -1;
        for (int i = 0; i < data.nRows; i++) {
            //+1 to skip the header
            String[] row = rawImport.get(i + 1);
            //iterate over the features
            for (int f = 0; f < data.nFeatures; f++) {
                String val = row[featCol[f]];
                if (data.isCategorical[f]) {
                    //get the category's running list of seen categories
                    List<String> cats = catValues.get(f);
                    //extract index in the running list the current observed category maps to
                    int code = cats.indexOf(val);
                    //extracted index < 0 -> not seen before, add to the running list
                    if (code < 0) {
                        code = cats.size();
                        cats.add(val);
                    }
                    //note int code stored as double
                    data.cols[f][i] = code;
                } else {
                    //numeric feat just added
                    data.cols[f][i] = Double.parseDouble(val);
                }
            }
            //add the target
            int target = Integer.parseInt(row[targetIdx]);
            data.y[i] = target;
            //default weight, bagging may change later
            data.weight[i] = 1.0;

            //track largest class id, needed for nClasses
            if (target > maxClass) {
                maxClass = target;
            } 
        }

        // categories[f] = the seen-categories list (index == code)
        for (int f = 0; f < data.nFeatures; f++)
            if (data.isCategorical[f])
                data.categories[f] = catValues.get(f).toArray(new String[0]);

        data.nClasses = maxClass + 1;
        return data;

    }

    // TODO: accessors as needed (e.g. categoryCount(f)); no missing-value handling required
    //       because service-missing is encoded as its own category upstream.
}

/**
 * One tree node. Internal nodes hold a split; leaves hold a class distribution.
 */
class Node {
    boolean isLeaf;

    //internal node
    int feature;            //feat index node splits on
    boolean numericSplit;   //true = binary threshold; false = multi-way categorical
    double threshold;       //used when numericSplit
    Node[] children;        //numeric: [left (<=threshold), right]; categorical: indexed by cat. code

    //leaf
    double[] classDist;     //weighted class distribution
    int predictedClass;           // argmax of classDist

}

/** Describes the chosen split at a node (one allocated per node, not per candidate). */
class Split {
    int feature;
    boolean numeric;
    double threshold;             // for numeric splits
    double score;                 // gain or gain ratio (depending on criterion)

}

/**
 * C4.5 classification tree: entropy / information gain / gain ratio split selection,
 * multi-way categorical + numeric threshold splits, REP pruning (optional).
 */
class C45Tree {
    enum Criterion {INFO_GAIN, GAIN_RATIO}

    /**
     * Helper method finds the majority class
    */
    static int argMax(double[] counts) {
        int bestIndex = 0;
        double bestValue = counts[0];
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] > bestValue) {
                bestValue = counts[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    /**
     * Helper method creates a leaf node
    */
    static Node makeLeaf(double[] counts) {
        Node leaf = new Node();
        leaf.isLeaf = true;
        leaf.classDist = counts;
        leaf.predictedClass = argMax(counts);
        return leaf;
    }

    // config
    Criterion criterion = Criterion.GAIN_RATIO;
    int minSamplesLeaf = 1;
    int maxDepth = Integer.MAX_VALUE;
    boolean prune = false;        // REP pruning for the single tree; false for forest base trees
    int maxFeatures = 0;          // 0 = consider all features; >0 = subspace sampling (forest)
    Random rng = new Random(42);

    //model
    Node root;
    int nClasses;

    /** Induce the tree from the given rows of the dataset. */
    void fit(Dataset d, int[] rows) {
        nClasses = d.nClasses;
        root = induce(d, rows, 0);
        //NB pruning still needs to be added later
    }

    /** Recursive induction (divide and conquer). */
    private Node induce(Dataset d, int[] rows, int depth) {

        double[] parentCounts = new double[d.nClasses];
        for (int k = 0; k < rows.length; k++) {
            int idx = rows[k];
            parentCounts[d.y[idx]] += d.weight[idx];
        }

        //First test if one of the three stoppitn rules reached:
        //1. All instances in the subset are the same class (pure / homogeneous).
        //2. Subset size falls below a threshold (too few instances to keep splitting).
        //3. Tree reaches a depth limit.
        boolean pure, tooFew, tooDeep;

        int classesPresent = 0;
        for (int c = 0; c < parentCounts.length; c++) {
            if (parentCounts[c] > 0) {
                classesPresent++;
            }
        }

        pure = (classesPresent == 1);
        tooFew = (rows.length < 2);
        tooDeep = (depth >= maxDepth);

        //If the stopping condition is reached then return the current node as a leaf
        if (pure || tooDeep || tooFew) {
            return makeLeaf(parentCounts);
        }

        //Now the split case
        Split split = bestSplit(d, rows);
        if (split.score <= 0) {
            //No useful split, node becomes a leaf
            return makeLeaf(parentCounts);
        }


        int f = split.feature;
        Node node = new Node();
        node.isLeaf = false;
        node.feature = f;

        if (split.numeric) {
            //NUMERIC -> two children
            double t = split.threshold;

            //First count the number of obs to the left + right of the split
            int leftCount = 0;                               
            for (int k = 0; k < rows.length; k++) {
                if (d.cols[f][rows[k]] <= t) {
                    leftCount++;
                }
            }

            //Then fill two arrays indicating observation indices to the left
            //and right of the split
            int[] leftRows  = new int[leftCount];
            int[] rightRows = new int[rows.length - leftCount];

            int li = 0;
            int ri = 0;
            for (int k = 0; k < rows.length; k++) {
                int idx = rows[k];
                if (d.cols[f][idx] <= t) {
                    leftRows[li] = idx;
                    li++;
                } else {
                    rightRows[ri] = idx;
                    ri++;
                }
            }

            node.numericSplit = true;
            node.threshold = t;
            node.children = new Node[2];
            //Recursive step, induce the two children either side of the split point
            node.children[0] = induce(d, leftRows,  depth + 1);
            node.children[1] = induce(d, rightRows, depth + 1);

        } else {
            //CATEGORICAL -> one node per category

            //# of categories
            int K = d.categories[f].length;

            //# observations per category
            int[] catCount = new int[K];                     // pass 1: count per category
            for (int k = 0; k < rows.length; k++) {
                int code = (int) d.cols[f][rows[k]];
                catCount[code]++;
            }

            //allocate arrays to store indexes of each category's observations
            int[][] catRows = new int[K][];
            for (int c = 0; c < K; c++) {
                catRows[c] = new int[catCount[c]];
            }
            int[] fillPos = new int[K];
            //fill the arrays
            for (int k = 0; k < rows.length; k++) {
                int idx = rows[k];
                int code = (int) d.cols[f][idx];
                catRows[code][fillPos[code]] = idx;
                fillPos[code]++;
            }

            node.numericSplit = false;
            //Child  node for each branch of the split
            node.children = new Node[K];
            for (int c = 0; c < K; c++) {
                if (catCount[c] == 0) {
                    // category absent here -> leaf predicting this node's majority
                    node.children[c] = makeLeaf(parentCounts);
                } else {
                    node.children[c] = induce(d, catRows[c], depth + 1);
                }
            }
        }

        //store the class info on internal nodes too so REP pruning can collapse them into leaves
        node.classDist = parentCounts;
        node.predictedClass = argMax(parentCounts);
        return node;
    }

    /** Select the best feature (and threshold if numeric) by gain / gain ratio. 
     *  note rows[] is an array of indices pointing to the observation in the current
     *  node
    */
    private Split bestSplit(Dataset d, int[] rows) {
        //array stores the feature and split point (if numeric) to split on
        Split optimalSplit = new Split();
        double optimalSplitMetric = Double.NEGATIVE_INFINITY;

        //store the weighted target class counts for the current node
        double[] parentCounts = new double[d.nClasses];
        for (int k = 0; k < rows.length; k++) {
            int idx = rows[k];
            parentCounts[d.y[idx]] += d.weight[idx];
        }

        //For the random forest we must choose a split from a list of randomly selected 
        //features
        int[] candidateFeatures;
        if (maxFeatures <= 0 || maxFeatures >= d.nFeatures) {
            //regular classification tree -> considerr all feats
            candidateFeatures = new int[d.nFeatures];
            for (int i = 0; i < d.nFeatures; i++) {
                candidateFeatures[i] = i;
            }
        } else {
            //randomly pick maxFeatures different features
            int[] pool = new int[d.nFeatures];
            for (int i = 0; i < d.nFeatures; i++) {
                pool[i] = i;
            }
            for (int i = 0; i < maxFeatures; i++) {
                //random index in from i to nFeatures-1
                int j = i + rng.nextInt(d.nFeatures - i);   
                int tmp = pool[i];
                pool[i] = pool[j];
                pool[j] = tmp;
            }
            candidateFeatures = new int[maxFeatures];
            for (int i = 0; i < maxFeatures; i++) {
                candidateFeatures[i] = pool[i];
            }
        }

        //iterate over all the candidates to find the best split
        for (int fi = 0; fi < candidateFeatures.length; fi++) {
            int f = candidateFeatures[fi];

            if (d.isCategorical[f] ==  true) {
                // multi-way split w/ one child per category
                int K = d.categories[f].length;

                //store weighted target class counts for the children nodes
                double[][] childCounts = new double[K][d.nClasses];
                //unweighted observation count per branch, for the minSamplesLeaf check
                int[] catSize = new int[K];
                for (int k = 0; k < rows.length; k++) {
                    int idx = rows[k];
                    int code = (int) d.cols[f][idx];          // code stored as double -> cast to int
                    childCounts[code][d.y[idx]] += d.weight[idx];
                    catSize[code]++;
                }

                //every non-empty branch must keep at least minSamplesLeaf observations (same guarantee as CART)
                boolean validSplit = true;
                for (int c = 0; c < K; c++) {
                    if (catSize[c] > 0 && catSize[c] < minSamplesLeaf) {
                        validSplit = false;
                        break;
                    }
                }
                if (validSplit == false) {
                    continue;
                }

                double score;
                if (criterion == Criterion.GAIN_RATIO) {
                    score = gainRatio(parentCounts, childCounts);
                } else {
                    score = informationGain(parentCounts, childCounts);
                }

                if (score > optimalSplitMetric) {
                    optimalSplitMetric = score;
                    optimalSplit.feature = f;
                    // categorical -> threshold unused
                    optimalSplit.numeric = false;             
                    optimalSplit.score   = score;
                }
            } else {
                //numeric feature
                int n = rows.length;

                // sort the node's row indices by this feature's value
                Integer[] order = new Integer[n];
                for (int k = 0; k < n; k++) {
                    order[k] = rows[k];
                }
                //lambda needs an effectively-final feature index
                final int featureIndex = f;   

                //use a comparator to sort the observations from small to large while keeping
                //track of the row indices
                Arrays.sort(order, (a, b) -> Double.compare(d.cols[featureIndex][a], d.cols[featureIndex][b]));

                // all rows start on the right; each step moves one row from right -> left
                double[] leftCounts  = new double[d.nClasses];
                double[] rightCounts = parentCounts.clone();

                for (int k = 0; k < n - 1; k++) {
                    int idx = order[k];
                    // move row idx
                    leftCounts[d.y[idx]]  += d.weight[idx];      
                    // from right to left
                    rightCounts[d.y[idx]] -= d.weight[idx];      

                    double valueAtK  = d.cols[f][order[k]];
                    double valueAtNext = d.cols[f][order[k + 1]];
                    if (valueAtK == valueAtNext) {
                        //if we have equal weights then it's not a real boundary
                        continue;
                    }

                    //both children must keep at least minSamplesLeaf observations (same guarantee as CART)
                    int leftSize = k + 1;
                    int rightSize = n - leftSize;
                    if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
                        continue;
                    }

                    double[][] childCounts = {leftCounts, rightCounts};
                    double score;
                    if (criterion == Criterion.GAIN_RATIO) {
                        score = gainRatio(parentCounts, childCounts);
                    } else {
                        score = informationGain(parentCounts, childCounts);
                    }

                    if (score > optimalSplitMetric) {
                        optimalSplitMetric     = score;
                        optimalSplit.feature   = f;
                        optimalSplit.numeric   = true;
                        optimalSplit.threshold = (valueAtK + valueAtNext) / 2.0;
                        optimalSplit.score     = score;
                    }
                }

            }
        }
        return optimalSplit;
    }

    /** Sum helper method*/
    static double sum(double[] a) {
        double s = 0;
        for (double x : a) {
            s += x;
        }
        return s;
    }


    /** calcs the entropy of a weighted class-count vector. */
    static double entropy(double[] classCounts) {
        //H = -sum_m p_m log p_m; choose log base m
        int numClasses = classCounts.length;
        double entropyOut = 0;
        double totalObs = 0;
        //get the total # of observations
        for (int i = 0; i < numClasses; i++) {
            totalObs += classCounts[i];
        }

        for (int i = 0; i < numClasses; i++) {
            if (classCounts[i] == 0) {
                //if 0 in the class skip because log(0) is -infinity
                continue;
            } else {
                double p =  classCounts[i]/totalObs;
                entropyOut -= p * Math.log(p);
            }
        }
        //Divide by Math.log(numClasses) to make the log base m
        return entropyOut / Math.log(numClasses);
    }

    /** Information gain = H(parent) - weighted sum of child entropies. */
    static double informationGain(double[] parentCounts, double[][] childCounts) {

        double totalChildCounts = 0;
        //stores the total # obs in the i'th child node
        double childICount[] = new double[childCounts.length];
        //count the total # obs amongst all of the children
        for (int i = 0; i < childCounts.length; i++) {
            childICount[i] = sum(childCounts[i]);
            totalChildCounts += childICount[i];
        }


        double parentEntropy = entropy(parentCounts);
        double weightedChildrenEntropy = 0;
        for (int i = 0; i < childCounts.length; i++) {
            //Weighted sum of child entropies
            weightedChildrenEntropy += (childICount[i]/totalChildCounts) * entropy(childCounts[i]);
        }

        return parentEntropy - weightedChildrenEntropy;
    }

    /** Split information = entropy of the child-size distribution. 
     * helps with the many-valued feature bias
    */
    static double splitInfo(double[][] childCounts) {

        //get the number of classes to normalise the log base
        double numClasses = childCounts[0].length;

        double totalChildCounts = 0;
        double childICount[] = new double[childCounts.length];
        //count the total # obs amongst all of the children
        for (int i = 0; i < childCounts.length; i++) {
            childICount[i] = sum(childCounts[i]);
            totalChildCounts += childICount[i];
        }

        double splitInfo = 0;
        for (int i = 0; i < childCounts.length; i++) {
            //skip empty children
            if (childICount[i] == 0) continue;
            double p = childICount[i]/totalChildCounts;
            splitInfo -= p * Math.log(p);
        }

        //Divide by log(numClasses) to make the log base m
        return splitInfo / Math.log(numClasses);
    }

    /** Gain ratio = information gain / split info. */
    static double gainRatio(double[] parentCounts, double[][] childCounts) {
    double si = splitInfo(childCounts);
        if (si > 0) {
            return informationGain(parentCounts, childCounts) / si;
        } else {
            //here we have a degenerate split -> no gain ratio
            return 0.0;
        }
    }


    /** Predict the class for a single row of a dataset. */
    int predict(Dataset d, int row) {
        
        Node current = root;
        //keep going till you reach a leaf
        while (!current.isLeaf) {
            //split feature
            int f = current.feature;

            if (current.numericSplit) {
                if (d.cols[f][row] <= current.threshold) {
                    //follow left split
                    current = current.children[0];
                } else {
                    //follow right split
                    current = current.children[1];
                }
            } else {
                //Categorical split

                //The class (code) of the row'th observation for the category on which 
                //the split happened
                int code = (int) d.cols[f][row];
                //follow this class branch
                current = current.children[code];
            }
            
        }
        //Return the predicted target label from the leaf node
        return current.predictedClass;
    }

    /** Reduced-error pruning (REP) against a held-out validation subset. */
    void repPrune(Dataset valid, int[] rows) {
        prune(root, valid, rows);
    }

    /** Prunes the subtree at node bottom-up; returns its validation-error count after pruning. */
    private double prune(Node node, Dataset valid, int[] rows) {
        //base case, a leaf's error count is just its misclassifications
        if (node.isLeaf) {
            return errorsAsLeaf(node, valid, rows);
        }

        int f = node.feature;
        double subtreeErrors = 0;

        if (node.numericSplit) {
            //route the validation rows left/right the same way predict does
            int leftCount = 0;
            for (int k = 0; k < rows.length; k++) {
                if (valid.cols[f][rows[k]] <= node.threshold) {
                    leftCount++;
                }
            }
            int[] leftRows  = new int[leftCount];
            int[] rightRows = new int[rows.length - leftCount];
            int li = 0;
            int ri = 0;
            for (int k = 0; k < rows.length; k++) {
                int idx = rows[k];
                if (valid.cols[f][idx] <= node.threshold) {
                    leftRows[li] = idx;
                    li++;
                } else {
                    rightRows[ri] = idx;
                    ri++;
                }
            }
            //recurse into the children first (bottom-up)
            subtreeErrors += prune(node.children[0], valid, leftRows);
            subtreeErrors += prune(node.children[1], valid, rightRows);
        } else {
            //route the validation rows into the matching category branch
            int K = node.children.length;
            int[] catCount = new int[K];
            for (int k = 0; k < rows.length; k++) {
                int code = (int) valid.cols[f][rows[k]];
                catCount[code]++;
            }
            int[][] catRows = new int[K][];
            for (int c = 0; c < K; c++) {
                catRows[c] = new int[catCount[c]];
            }
            int[] fillPos = new int[K];
            for (int k = 0; k < rows.length; k++) {
                int idx = rows[k];
                int code = (int) valid.cols[f][idx];
                catRows[code][fillPos[code]] = idx;
                fillPos[code]++;
            }
            for (int c = 0; c < K; c++) {
                subtreeErrors += prune(node.children[c], valid, catRows[c]);
            }
        }

        //errors if we instead collapse this node into a single leaf
        double leafErrors = errorsAsLeaf(node, valid, rows);

        if (leafErrors <= subtreeErrors) {
            //the leaf is no worse so prune (ties favour the simpler tree)
            node.isLeaf = true;
            node.children = null;
            return leafErrors;
        } else {
            return subtreeErrors;
        }
    }

    /** Validation errors if node predicts its stored majority class for all rows. */
    private double errorsAsLeaf(Node node, Dataset valid, int[] rows) {
        double errors = 0;
        for (int k = 0; k < rows.length; k++) {
            if (valid.y[rows[k]] != node.predictedClass) {
                errors += valid.weight[rows[k]];
            }
        }
        return errors;
    }
}

/**
 * Bagging + feature-subspace sampling over C45Tree (unpruned base trees).
 * Provides out-of-bag error so the forest can be tuned without nested CV.
 */
class RandomForest {
    //standard cofiguration
    int nTrees = 200;
    //passed to each C45Tree (if >0 enables subspace sampling)
    int maxFeatures = 0;          
    C45Tree.Criterion criterion = C45Tree.Criterion.GAIN_RATIO;
    int minSamplesLeaf = 1;
    Random rng = new Random(42);

    //Model
    C45Tree[] trees;
    int nClasses;
    //inBag[t][i] = true if row i was in tree t's bootstrap sample
    boolean[][] inBag;   

    /** Fit the forest: bootstrap sample per tree, train each tree. */
    void fit(Dataset d) {

        int numSamples = d.nRows;
        nClasses = d.nClasses;
        trees = new C45Tree[nTrees];
        inBag = new boolean[nTrees][numSamples];
        //Iterate over all of the trees
        for (int t = 0; t < nTrees; t++) {
            //bootstrap sample
            int[] booty = new int[numSamples];
            for (int i = 0; i < numSamples; i++) {
                booty[i] = rng.nextInt(numSamples);
                //mark in bag obs
                inBag[t][booty[i]] = true;
            }

            //configure tree and build on the bootstrap sample
            C45Tree tree = new C45Tree();
            tree.criterion = criterion;
            tree.minSamplesLeaf = minSamplesLeaf;
            tree.maxFeatures = maxFeatures;
            tree.prune = false;
            tree.rng = new Random(rng.nextLong());

            tree.fit(d, booty);
            trees[t] = tree;
        }
    }

    /** Majority-vote across all trees. */
    int predict(Dataset d, int row) {

        double[] votes = new double[nClasses];
        for (int t = 0; t < nTrees; t++) {
            int classs = trees[t].predict(d, row);
            votes[classs] += 1.0;
        }
        //class with most votes
        return C45Tree.argMax(votes);
    }

    /** OOB macro-F1 generalisation estimate we use for tuning 
     * for each row vote using only trees where the obs was OOB
    */
    double oobMacroF1(Dataset d) {
        int numSamples = d.nRows;
        // OOB-voted prediction for each observation
        int[] oobPred = new int[numSamples];
        //keep track of if the observation has an OOB tree
        boolean[] hasOob = new boolean[numSamples];   


        for (int i = 0; i < numSamples; i++) {
            double[] votes = new double[nClasses];
            int votingTrees = 0;
            for (int t = 0; t < nTrees; t++) {
                if (!inBag[t][i]) { 
                    //tree t didn't train on row i
                    int classs = trees[t].predict(d, i);
                    votes[classs] += 1.0;
                    votingTrees++;
                }
            }
            if (votingTrees > 0) {
                oobPred[i] = C45Tree.argMax(votes);
                hasOob[i] = true;
            }
        }

        // collect only the rows that got an OOB prediction then score them
        int count = 0;
        for (int i = 0; i < numSamples; i++) {
            if (hasOob[i]) {
                count++;
            }
        }
        int[] yTrue = new int[count];
        int[] yPred = new int[count];
        int k = 0;
        for (int i = 0; i < numSamples; i++) {
            if (hasOob[i]) {
                yTrue[k] = d.y[i];
                yPred[k] = oobPred[i];
                k++;
            }
        }
        // calc the macro f1 using the method we defined
        return Metrics.macroF1(yTrue, yPred, nClasses);
    }

}

/** Metrics for Java-side tuning selection only (final reported metrics computed in the notebook). */
class Metrics {
    //Macro-averaged F1 over nClasses
    static double macroF1(int[] yTrue, int[] yPred, int nClasses) {
        int[] tp = new int[nClasses];   //true positives
        int[] predicted = new int[nClasses]; //# obs predicted as class c
        int[] actual = new int[nClasses]; //# obs actually are class c

        //build up true and predicted arrays
        for (int i = 0; i < yTrue.length; i++) {
            int t = yTrue[i];
            int p = yPred[i];
            actual[t]++;
            predicted[p]++;
            if (t == p) {
                tp[t]++;
            }
        }


        double sumF1 = 0.0;
        for (int c = 0; c < nClasses; c++) {
            double precision;
            if (predicted[c] == 0) {
                //nothing predicted as c, 0 precision
                precision = 0.0;
            } else {
                precision = (double) tp[c] / predicted[c];
            }

            double recall;
            if (actual[c] == 0) {
                //no true c, 0 recall
                recall = 0.0;
            } else {
                recall = (double) tp[c]/actual[c];
            }

            double f1;
            if (precision + recall == 0.0) {
                //set 0 to avoid division by 0
                f1 = 0.0;
            } else {
                f1 = 2.0 * precision * recall/(precision + recall);
            }

            sumF1 += f1;
        }
        //note macro f1, unweighted mean over all classes
        return sumF1 / nClasses;
    }

}

/** Minimal CSV read / write. */
class CSV {
    //Read in every row from the csv as a list of strings
    static List<String[]> read(String path) {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(path))) {
            String line = br.readLine();
            while (line != null) {
                if (!line.isEmpty()) {
                    //-1 ensures missing values are added if theyre in the last field
                    rows.add(line.split(",", -1));
                }
                //Read the next line
                line = br.readLine();
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Couldn't read the csv " + path, e);
        }
        return rows;
    }    

        //       writePredictions(path, rowIds, folds, preds) for the notebook to score.

    //Write out-of-fold predictions for the notebook to score: columns row_id,fold,prediction
    static void writePredictions(String path, int[] rowIds, int[] folds, int[] preds) {
        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(path))) {
            bw.write("row_id,fold,prediction");
            bw.newLine();
            for (int i = 0; i < rowIds.length; i++) {
                bw.write(rowIds[i] + "," + folds[i] + "," + preds[i]);
                bw.newLine();
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Couldn't write predictions to " + path, e);
        }
    }

}
