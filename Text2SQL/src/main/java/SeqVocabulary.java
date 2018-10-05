import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.examples.nlp.sequencevectors.SequenceVectorsTextExample;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;

public class SeqVocabulary {

    private static Logger log = LoggerFactory.getLogger(PartOfSpeech.class);

    public static void main(String[] args) throws Exception {

        /*

         **************SEQUENCE VOCABULARY*********************

         */

            //Carregar CSV

        File inputFile = new ClassPathResource("Exemplos_de_Query_1.csv").getFile();

        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(inputFile));

        SentenceIterator iter = new BasicLineIterator(inputFile);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        //First we build line iterator
        BasicLineIterator underlyingIterator = new BasicLineIterator(inputFile);

        //Now we need the way to convert lines into Sequences of VocabWords.
        //In this example that's SentenceTransformer


        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();

        //And we pack that transformer into AbstractSequenceIterator

        AbstractSequenceIterator<VocabWord> sequenceIterator =
                new AbstractSequenceIterator.Builder<>(transformer).build();

        //Now we should build vocabulary out of sequence iterator.
        //We can skip this phase, and just set AbstractVectors.resetModel(TRUE), and vocabulary will be mastered internally

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 5)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);

        //Time to build WeightLookupTable instance for our new model

        WeightLookupTable<VocabWord> lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(150)
                .useAdaGrad(false)
                .cache(vocabCache)
                .build();

        //reset model is viable only if you're setting AbstractVectors.resetModel() to false
        //if set to True - it will be called internally

        lookupTable.resetWeights(true);

        //Now we can build AbstractVectors model, that suits our needs

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                // minimum number of occurencies for each element in training corpus. All elements below this value will be ignored
                // Please note: this value has effect only if resetModel() set to TRUE, for internal model building. Otherwise it'll be ignored, and actual vocabulary content will be used
                .minWordFrequency(5)

                // WeightLookupTable
                .lookupTable(lookupTable)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // vocabulary built prior to modelling
                .vocabCache(vocabCache)

                // batchSize is the number of sequences being processed by 1 thread at once
                // this value actually matters if you have iterations > 1
                .batchSize(250)

                // number of iterations over batch
                .iterations(1)

                // number of iterations over whole training corpus
                .epochs(50)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(false)


                //These two methods define our training goals. At least one goal should be set to TRUE.

                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)

                //Specifies elements learning algorithms. SkipGram, for example.

                .elementsLearningAlgorithm(new SkipGram<VocabWord>())

                .build();

        //Now, after all options are set, we just call fit()

        vectors.fit();

        /*
            As soon as fit() exits, model considered built, and we can test it.
            Please note: all similarity context is handled via SequenceElement's labels, so if you're using AbstractVectors to build models for complex
            objects/relations please take care of Labels uniqueness and meaning for yourself.
         */
        double sim = vectors.similarity("show", "select");
        log.info("show/findt similarity: " + sim);

        log.info("\n\n....................Closest Words:...................");
        Collection<String> lst2 = vectors.wordsNearestSum("show", 10);
        log.info("10 Words closest to 'Show': {}", lst2);

        log.info("\n\n....................Closest Words:...................");
        Collection<String> lst3 = vectors.wordsNearestSum("select", 10);
        log.info("10 Words closest to 'select': {}", lst3);
    }
}
