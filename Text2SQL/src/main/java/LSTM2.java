import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.examples.recurrent.encdec.CorpusIterator;
import org.deeplearning4j.examples.recurrent.encdec.CorpusProcessor;
import org.deeplearning4j.examples.recurrent.encdec.EncoderDecoderLSTM;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class LSTM2 {

    /**
     * Dictionary that maps words into numbers.
     */
    private final Map<String, Double> dict = new HashMap<>();

    /**
     * Reverse map of {@link #dict}.
     */
    private final Map<Double, String> revDict = new HashMap<>();

    private final String CHARS = "-\\/_&" + CorpusProcessor.SPECIALS;

    /**
     * The contents of the corpus. This is a list of sentences (each word of the
     * sentence is denoted by a {@link java.lang.Double}).
     */
    private final List<List<Double>> corpus = new ArrayList<>();

    private static final int HIDDEN_LAYER_WIDTH = 512; // this is purely empirical, affects performance and VRAM requirement
    private static final int EMBEDDING_WIDTH = 128; // one-hot vectors will be embedded to more dense vectors with this width
    private static final String CORPUS_FILENAME = "Exemplos_de_Query_1.csv"; // filename of data corpus to learn
    private static final String MODEL_FILENAME = "rnn_train.zip"; // filename of the model
    private static final String BACKUP_MODEL_FILENAME = "rnn_train.bak.zip"; // filename of the previous version of the model (backup)
    private static final int MINIBATCH_SIZE = 32;
    private static final Random rnd = new Random(new Date().getTime());
    private static final long SAVE_EACH_MS = TimeUnit.MINUTES.toMillis(5); // save the model with this period
    private static final long TEST_EACH_MS = TimeUnit.MINUTES.toMillis(1); // test the model with this period
    private static final int MAX_DICT = 20000; // this number of most frequent words will be used, unknown words (that are not in the
    // dictionary) are replaced with <unk> token
    private static final int TBPTT_SIZE = 25;
    private static final double LEARNING_RATE = 1e-1;
    private static final double RMS_DECAY = 0.95;
    private static final int ROW_SIZE = 40; // maximum line length in tokens

    /**
     * The delay between invocations of {@link java.lang.System#gc()} in
     * milliseconds. If VRAM is being exhausted, reduce this value. Increase
     * this value to yield better performance.
     */
    private static final int GC_WINDOW = 2000;

    private static final int MACROBATCH_SIZE = 20; // see CorpusIterator

    /**
     * The computation graph model.
     */
    private ComputationGraph net;

    public static void main(String[] args) throws IOException {
        new LSTM2().run(args);
    }

    private void run(String[] args) throws IOException {
        Nd4j.getMemoryManager().setAutoGcWindow(GC_WINDOW);

        createDictionary();

        File networkFile = new File(toTempPath(MODEL_FILENAME));
        int offset = 0;
        if (networkFile.exists()) {
            System.out.println("Loading the existing network...");
            net = ModelSerializer.restoreComputationGraph(networkFile);
            System.out.print("Enter d to start dialog or a number to continue training from that minibatch: ");
            String input;
            try (Scanner scanner = new Scanner(System.in)) {
                input = scanner.nextLine();
                if (input.toLowerCase().equals("d")) {
                    startDialog(scanner);
                } else {
                    offset = Integer.valueOf(input);
                    test();
                }
            }
        } else {
            System.out.println("Creating a new network...");
            createComputationGraph();
        }
        System.out.println("Number of parameters: " + net.numParams());
        net.setListeners(new ScoreIterationListener(1));
        train(networkFile, offset);
    }

    /**
     * Configure and initialize the computation graph. This is done once in the
     * beginning to prepare the {@link #net} for training.
     */
    private void createComputationGraph() {
        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .pretrain(false)
                .backprop(true)
                .backpropType(BackpropType.Standard)
                .tBPTTBackwardLength(TBPTT_SIZE)
                .tBPTTForwardLength(TBPTT_SIZE)
                .addInputs("inputLine", "decoderInput")
                .setInputTypes(InputType.recurrent(dict.size()), InputType.recurrent(dict.size()))
                .addLayer("embeddingEncoder",
                        new EmbeddingLayer.Builder()
                                .nIn(dict.size())
                                .nOut(EMBEDDING_WIDTH)
                                .build(),
                        "inputLine")
                .addLayer("encoder",
                        new LSTM.Builder()
                                .nIn(EMBEDDING_WIDTH)
                                .nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.TANH)
                                .build(),
                        "embeddingEncoder")
                .addVertex("thoughtVector", new LastTimeStepVertex("inputLine"), "encoder")
                .addVertex("dup", new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
                .addVertex("merge", new MergeVertex(), "decoderInput", "dup")
                .addLayer("decoder",
                        new LSTM.Builder()
                                .nIn(dict.size() + HIDDEN_LAYER_WIDTH)
                                .nOut(HIDDEN_LAYER_WIDTH)
                                .activation(Activation.TANH)
                                .build(),
                        "merge")
                .addLayer("output",
                        new RnnOutputLayer.Builder()
                                .nIn(HIDDEN_LAYER_WIDTH)
                                .nOut(dict.size())
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "decoder")
                .setOutputs("output");

        net = new ComputationGraph(graphBuilder.build());
        net.init();
    }

    private void train(File networkFile, int offset) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        long lastTestTime = System.currentTimeMillis();
        CorpusIterator logsIterator = new CorpusIterator(corpus, MINIBATCH_SIZE, MACROBATCH_SIZE, dict.size(), ROW_SIZE);
        for (int epoch = 1; epoch < 10000; ++epoch) {
            System.out.println("Epoch " + epoch);
            if (epoch == 1) {
                logsIterator.setCurrentBatch(offset);
            } else {
                logsIterator.reset();
            }
            int lastPerc = 0;
            while (logsIterator.hasNextMacrobatch()) {
                net.fit(logsIterator);
                logsIterator.nextMacroBatch();
                System.out.println("Batch = " + logsIterator.batch());
                int newPerc = (logsIterator.batch() * 100 / logsIterator.totalBatches());
                if (newPerc != lastPerc) {
                    System.out.println("Epoch complete: " + newPerc + "%");
                    lastPerc = newPerc;
                }
                if (System.currentTimeMillis() - lastSaveTime > SAVE_EACH_MS) {
                    saveModel(networkFile);
                    lastSaveTime = System.currentTimeMillis();
                }
                if (System.currentTimeMillis() - lastTestTime > TEST_EACH_MS) {
                    test();
                    lastTestTime = System.currentTimeMillis();
                }
            }
        }
    }

    private void startDialog(Scanner scanner) throws IOException {
        System.out.println("Dialog started.");
        while (true) {
            System.out.print("In> ");
            // input line is appended to conform to the corpus format
            String line = "1 +++$+++ u11 +++$+++ m0 +++$+++ WALTER +++$+++ " + scanner.nextLine() + "\n";
            CorpusProcessor dialogProcessor = new CorpusProcessor(new ByteArrayInputStream(line.getBytes(StandardCharsets.UTF_8)), ROW_SIZE,
                    false) {
                @Override
                protected void processLine(String lastLine) {
                    List<String> words = new ArrayList<>();
                    tokenizeLine(lastLine, words, true);
                    final List<Double> wordIdxs = wordsToIndexes(words);
                    if (!wordIdxs.isEmpty()) {
                        System.out.print("Got words: ");
                        for (Double idx : wordIdxs) {
                            System.out.print(revDict.get(idx) + " ");
                        }
                        System.out.println();
                        System.out.print("Out> ");
                        output(wordIdxs, true);
                    }
                }
            };
            dialogProcessor.setDict(dict);
            dialogProcessor.start();
        }
    }

    private void saveModel(File networkFile) throws IOException {
        System.out.println("Saving the model...");
        File backup = new File(toTempPath(BACKUP_MODEL_FILENAME));
        if (networkFile.exists()) {
            if (backup.exists()) {
                backup.delete();
            }
            networkFile.renameTo(backup);
        }
        ModelSerializer.writeModel(net, networkFile, true);
        System.out.println("Done.");
    }

    private void test() {
        System.out.println("======================== TEST ========================");
        int selected = rnd.nextInt(corpus.size());
        List<Double> rowIn = new ArrayList<>(corpus.get(selected));
        System.out.print("In: ");
        for (Double idx : rowIn) {
            System.out.print(revDict.get(idx) + " ");
        }
        System.out.println();
        System.out.print("Out: ");
        output(rowIn, true);
        System.out.println("====================== TEST END ======================");
    }

    private void output(List<Double> rowIn, boolean printUnknowns) {
        net.rnnClearPreviousState();
        Collections.reverse(rowIn);
        INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
        double[] decodeArr = new double[dict.size()];
        decodeArr[2] = 1;
        INDArray decode = Nd4j.create(decodeArr, new int[] { 1, dict.size(), 1 });
        net.feedForward(new INDArray[] { in, decode }, false, false);
        org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) net
                .getLayer("decoder");
        Layer output = net.getLayer("output");
        GraphVertex mergeVertex = net.getVertex("merge");
        INDArray thoughtVector = mergeVertex.getInputs()[1];
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
        for (int row = 0; row < ROW_SIZE; ++row) {
            mergeVertex.setInputs(decode, thoughtVector);
            INDArray merged = mergeVertex.doForward(false, mgr);
            INDArray activateDec = decoder.rnnTimeStep(merged, mgr);
            INDArray out = output.activate(activateDec, false, mgr);
            double d = rnd.nextDouble();
            double sum = 0.0;
            int idx = -1;
            for (int s = 0; s < out.size(1); s++) {
                sum += out.getDouble(0, s, 0);
                if (d <= sum) {
                    idx = s;
                    if (printUnknowns || s != 0) {
                        System.out.print(revDict.get((double) s) + " ");
                    }
                    break;
                }
            }
            if (idx == 1) {
                break;
            }
            double[] newDecodeArr = new double[dict.size()];
            newDecodeArr[idx] = 1;
            decode = Nd4j.create(newDecodeArr, new int[] { 1, dict.size(), 1 });
        }
        System.out.println();
    }

    private void createDictionary() throws IOException, FileNotFoundException {
        double idx = 3.0;
        dict.put("<unk>", 0.0);
        revDict.put(0.0, "<unk>");
        dict.put("<eos>", 1.0);
        revDict.put(1.0, "<eos>");
        dict.put("<go>", 2.0);
        revDict.put(2.0, "<go>");
        for (char c : CHARS.toCharArray()) {
            if (!dict.containsKey(c)) {
                dict.put(String.valueOf(c), idx);
                revDict.put(idx, String.valueOf(c));
                ++idx;
            }
        }
        System.out.println("Building the dictionary...");
        CorpusProcessor corpusProcessor = new CorpusProcessor(toTempPath(CORPUS_FILENAME), ROW_SIZE, true);
        corpusProcessor.start();
        Map<String, Double> freqs = corpusProcessor.getFreq();
        Set<String> dictSet = new TreeSet<>(); // the tokens order is preserved for TreeSet
        Map<Double, Set<String>> freqMap = new TreeMap<>(new Comparator<Double>() {

            @Override
            public int compare(Double o1, Double o2) {
                return (int) (o2 - o1);
            }
        }); // tokens of the same frequency fall under the same key, the order is reversed so the most frequent tokens go first
        for (Map.Entry<String, Double> entry : freqs.entrySet()) {
            Set<String> set = freqMap.get(entry.getValue());
            if (set == null) {
                set = new TreeSet<>(); // tokens of the same frequency would be sorted alphabetically
                freqMap.put(entry.getValue(), set);
            }
            set.add(entry.getKey());
        }
        int cnt = 0;
        dictSet.addAll(dict.keySet());
        // get most frequent tokens and put them to dictSet
        for (Map.Entry<Double, Set<String>> entry : freqMap.entrySet()) {
            for (String val : entry.getValue()) {
                if (dictSet.add(val) && ++cnt >= MAX_DICT) {
                    break;
                }
            }
            if (cnt >= MAX_DICT) {
                break;
            }
        }
        // all of the above means that the dictionary with the same MAX_DICT constraint and made from the same source file will always be
        // the same, the tokens always correspond to the same number so we don't need to save/restore the dictionary
        System.out.println("Dictionary is ready, size is " + dictSet.size());
        // index the dictionary and build the reverse dictionary for lookups
        for (String word : dictSet) {
            if (!dict.containsKey(word)) {
                dict.put(word, idx);
                revDict.put(idx, word);
                ++idx;
            }
        }
        System.out.println("Total dictionary size is " + dict.size() + ". Processing the dataset...");
        corpusProcessor = new CorpusProcessor(toTempPath(CORPUS_FILENAME), ROW_SIZE, false) {
            @Override
            protected void processLine(String lastLine) {
                List<String> words = new ArrayList<>();
                tokenizeLine(lastLine, words, true);
                corpus.add(wordsToIndexes(words));
            }
        };
        corpusProcessor.setDict(dict);
        corpusProcessor.start();
        System.out.println("Done. Corpus size is " + corpus.size());
    }

    private String toTempPath(String path) {
        return System.getProperty("java.io.tmpdir") + "/" + path;
    }

}
