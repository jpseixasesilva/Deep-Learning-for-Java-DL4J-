import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.hdf5;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class LSTM {
    public static void main(String[] args) throws Exception {

        //Read the novel.txt and convert to String
        //File inputFile = new ClassPathResource("Lusíadas_Camões.txt").getFile();
        //String inputData = IOUtils.toString(new FileInputStream("novel.txt"), "UTF-8");
        File inputFile = new ClassPathResource("Exemplos_de_Query_1.csv").getFile();
        String inputData = IOUtils.toString(new FileInputStream(inputFile));
        //Using only the first 50000 characters to speed up
        inputData = inputData.substring(0, 50000);

        //The alphabet we will using
        String validCharacters =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890\"\n',.?;()[]{}:!- ";

        //Create GraveLSTM  as input layer and RnnOutPutLayer as output
        GravesLSTM.Builder lstmBuilder = new GravesLSTM.Builder();
        lstmBuilder.activation(Activation.TANH);
        lstmBuilder.nIn(validCharacters.length());
        lstmBuilder.nOut(30); // Hidden
        GravesLSTM inputLayer = lstmBuilder.build();

        //Create RnnOutputLayer
        RnnOutputLayer.Builder outputBuilder = new RnnOutputLayer.Builder();
        outputBuilder.lossFunction(LossFunctions.LossFunction.MSE);
        outputBuilder.activation(Activation.SOFTMAX);
        outputBuilder.nIn(30); // Hidden
        outputBuilder.nOut(validCharacters.length());
        RnnOutputLayer outputLayer = outputBuilder.build();

        //Create MultiLayerNetwork to neuron layers to be ready
        NeuralNetConfiguration.Builder nnBuilder = new NeuralNetConfiguration.Builder();
        nnBuilder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        nnBuilder.updater(Updater.ADAM);
        nnBuilder.weightInit(WeightInit.XAVIER);
        nnBuilder.updater(new Nesterovs(0.01));
        nnBuilder.miniBatch(true);

        MultiLayerNetwork network = new MultiLayerNetwork(
                nnBuilder.list().layer(0, inputLayer)
                        .layer(1, outputLayer)
                        .backprop(true).pretrain(false)
                        .build());

        network.init();

        //Training data inside INDArray objects
        INDArray inputArray = Nd4j.zeros(1, inputLayer.getNIn(), inputData.length());
        INDArray inputLabels = Nd4j.zeros(1, outputLayer.getNOut(), inputData.length());


        //Create dataset that can be using in the NN

        //DataSet dataSet = new DataSet(inputArray, inputLabels);
       // DataSet dataSet = new DataSet(inputArray, inputLabels);








    }
}
