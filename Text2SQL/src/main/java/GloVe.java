import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

public class GloVe {

    /*
     **********GLOVE FITTING**********
     */
    private static Logger log = LoggerFactory.getLogger(GloVe.class);

    public static void main(String[] args) throws Exception {

        File inputFile = new ClassPathResource("Exemplos_de_Query_1.csv").getFile();

        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(inputFile));

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        //Glove glove = new Glove.Builder()
        Glove glove = new Glove.Builder()
                .iterate(iter)
                .tokenizerFactory(t)
                .alpha(0.75)
                .learningRate(0.1)
                .epochs(25)
                // cutoff for weighting function
                .xMax(100)
                // training is done in batches taken from training corpus
                .batchSize(1000)
                // if set to true, batches will be shuffled before training
                .shuffle(true)
                // if set to true word pairs will be built in both directions, LTR and RTL
                .symmetric(true)
                .build();

        glove.fit();

        double simD1 = glove.similarity("day", "night");
        log.info("Day/night similarity: " + simD1);

        Collection<String> words = glove.wordsNearest("day", 10);
        log.info("Nearest words to 'day': " + words);


        /*Collection<String> kingList = vec.wordsNearest(Arrays.asList("king", "woman"), Arrays.asList("queen"), 10);

        WeightLookupTable weightLookupTable = vec.lookupTable();
        Iterator<INDArray> vectors = weightLookupTable.vectors();
        INDArray wordVectorMatrix = vec.getWordVectorMatrix("myword");
        double[] wordVector = vec.getWordVector("myword");
        */

    }

}
