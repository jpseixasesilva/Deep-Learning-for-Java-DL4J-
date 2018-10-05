import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.Schema.Builder;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.examples.recurrent.encdec.EncoderDecoderLSTM;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class PartOfSpeech {

    private static Logger log = LoggerFactory.getLogger(PartOfSpeech.class);

    //public static void main() throws IOException {

    //}

    public static void main(String[] args) throws Exception {

        //Carregar CSV

        File inputFile = new ClassPathResource("Exemplos_de_Query_1.csv").getFile();

        Schema inputDataSchema = new Builder()
                .addColumnsString("InputText", "MySQL_Output", "PostgresSQL_Output")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("MySQL_Output", "PostgresSQL_Output")
                .build();

        Schema outputSchema = tp.getFinalSchema();

        File outputFile = new File("Output_File.csv");
        if(outputFile.exists()){
            outputFile.delete();
        }
        outputFile.createNewFile();

        //Define input reader and output writer:
        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(inputFile));

        RecordWriter rw = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);

        //Process the data:
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        rw.writeBatch(processedData);
        rw.close();

        System.out.print(inputDataSchema);
        String originalFileContents = FileUtils.readFileToString(outputFile);
        System.out.println(originalFileContents);

        System.out.printf("\n\n\n\n ----------- WORD2VEC --------- \n\n\n\n");
        log.info("Load and Vectorize Input Text");

        SentenceIterator iter = new BasicLineIterator(inputFile);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building Model...");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(20)
                .iterations(3)
                .layerSize(500)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("\n\n....................Fitting Word2Vec Model............................");
        vec.fit();

        log.info("\n\n....................Closest Words:...................");
        Collection<String> lst = vec.wordsNearestSum("show", 10);
        log.info("10 Words closest to 'Show': {}", lst);

//        Collection<String> lst1 = vec.wordsNearestSum("find", 10);
  //      log.info("10 Words closest to 'Find': {}", lst1);

        log.info("\n\n..................Save vectors...............");
        WordVectorSerializer.writeWord2VecModel(vec, "/home/joaopauloseixas/Documentos/DL4J/Text2SQL/src/main/resources/pathToSaveModel.txt");

        double simD = vec.similarity("show", "select");
        log.info("\n\n-----Similaridade entre as palavras 'SHOW' e 'FIND-----': "+simD);





        //EncoderDecoderLSTM encoderDecoderLSTM = new














    }

}