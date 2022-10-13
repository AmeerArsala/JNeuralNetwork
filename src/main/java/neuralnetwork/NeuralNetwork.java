package neuralnetwork;

import math.Tensor;
import neuralnetwork.training.LearningAlgorithm;
import neuralnetwork.training.NetworkParams;
import neuralnetwork.training.TrainingExample;
import neuralnetwork.util.MechIndex;
import neuralnetwork.util.MechNetworkIndex;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

public class NeuralNetwork {
    private final Layer[] layers;

    public NeuralNetwork(Layer[] layers) {
        this.layers = layers;
        reset(); //initialize weights and biases
    }

    public NeuralNetwork(int[] sizes) {
        layers = new Layer[sizes.length];

        int prevSize = 1; // only 1 needed for input layer
        for (int i = 0; i < sizes.length; ++i) {
            int size = sizes[i];
            layers[i] = new Layer(size, prevSize);
            prevSize = size;
        }

        reset(); //initialize weights and biases
    }

    //For targeting individual neurons in the network
    public NeuralNetwork setMechanics(MechNetworkIndex... mechs) {
        for (MechNetworkIndex mechI : mechs) {
            layers[mechI.layer].setMechanics(mechI.mechIndex);
        }

        return this;
    }

    //For targeting entire layers in the network
    public NeuralNetwork setDenseMechanics(MechIndex... mechs) {
        for (MechIndex mechIndex : mechs) {
            layers[mechIndex.i].setDenseMechanics(mechIndex.mechanics);
        }

        return this;
    }

    //randomize w and b everywhere
    public void reset() {
        NetworkParams networkParams = getNetworkParams(); // get current params of ANN, specifically the "shape" matters here
        DoubleUnaryOperator randomization = (theta) -> { return Math.random(); }; // reset operation

        layers[0].zeroWeights();

        setNetworkParams(networkParams.applyEntrywise(randomization), 1); // not input layer, already has weights set to 0

    }

    public int getNumLayers() { return layers.length; }

    public Layer getLayer(int i) {
        return layers[i];
    }

    public Layer getInputLayer() {
        return layers[0];
    }

    public Layer getOutputLayer() {
        return layers[layers.length - 1];
    }

    public NetworkParams getNetworkParams() { //relevant to the shape of the network
        Tensor T_W = new Tensor(layers.length), T_b = new Tensor(layers.length);
        for (int i = 0; i < layers.length; i++) {
            SimpleMatrix W = layers[i].getWeights().copy();
            SimpleMatrix b = layers[i].getBiases().copy();

            T_W.set(i, W);
            T_b.set(i, b);
        }

        return new NetworkParams(T_W, T_b);
    }

    public void setNetworkParams(NetworkParams netParams) {
        setNetworkParams(netParams, 0);
    }

    public void setNetworkParams(NetworkParams netParams, int startLayer) {
        for (int l = startLayer; l < layers.length; l++) {
            List<Neuron> neurons = layers[l].getNeurons();
            SimpleMatrix W_l = netParams.TW.get(l);
            SimpleMatrix b_l = netParams.Tb.get(l);
            for (int i = 0, len = neurons.size(); i < len; i++) {
                Neuron neuron = neurons.get(i);
                neuron.weights = W_l.extractVector(true, i);
                neuron.bias = b_l.get(i);
            }
        }
    }

    public void train(List<TrainingExample> allTrainingExamples, LearningAlgorithm learningAlgorithm) {
        NetworkParams currentNetworkParams = getNetworkParams();
        learningAlgorithm.init(allTrainingExamples, currentNetworkParams.skeleton());

        do {
            /*NetworkParams gradient = calculateGradient(learningAlgorithm.shuffleData());
            NetworkParams nextNetParams = learningAlgorithm.learnStep(currentNetworkParams, gradient);*/
            NetworkParams nextNetParams = learningAlgorithm.learnStep(this, currentNetworkParams);
            setNetworkParams(nextNetParams);

            System.err.println("NEW PARAMS: " + nextNetParams);
        } while (!learningAlgorithm.doesConverge());
    }

    public SimpleMatrix predict(double[] X) {
        Tensor allActivations = predictWithAllStats(X);
        return allActivations.getLast(); //only return the last layer
    }

    public Tensor predictWithAllStats(double[] X) {
        Tensor allActivations = new Tensor(layers.length); //record data of activations

        Layer currentLayer = layers[0]; //current layer is input layer

        currentLayer.zeroWeights(); // weights are 0 in the input layer
        currentLayer.setBiases(X);  // inputs in the input layer

        //Forward Propagation
        SimpleMatrix activations = currentLayer.getBiases(); //might seem redundant but it's for ANN stats
        allActivations.set(0, activations.copy()); //data recording step

        for (int i = 1; i < layers.length; i++) {
            currentLayer = layers[i];
            activations = currentLayer.activations(activations); // a' = Activations(Wa + b)

            allActivations.set(i, activations.copy()); //data recording step
        }

        //System.err.println("Layer-wise params: \n" + getNetworkParams().toString());
        //System.err.println("Layer-wise activations: \n" + allActivations);
        return allActivations;
    }

    public SimpleMatrix fastPredict(double[] X) { //doesn't record data
        Layer currentLayer = layers[0]; //current layer is input layer

        currentLayer.zeroWeights();
        currentLayer.setBiases(X);

        //Forward Propagation
        SimpleMatrix activations = Operations.colVector(X);

        for (int i = 1; i < layers.length; i++) {
            currentLayer = layers[i];
            activations = currentLayer.activations(activations); // a' = Activations(Wa + b)
        }

        return activations;
    }
}
