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
        learningAlgorithm.init(allTrainingExamples);
        NetworkParams currentNetworkParams = getNetworkParams();

        do {
            NetworkParams gradient = calculateGradient(learningAlgorithm.shuffleData());
            NetworkParams nextNetParams = learningAlgorithm.learnStep(currentNetworkParams, gradient);
            setNetworkParams(nextNetParams);
        } while (!learningAlgorithm.doesConverge());
    }

    private NetworkParams calculateGradient(List<TrainingExample> trainingExamples) {
        //List<NetworkParams> allGradients = new LinkedList<>(); //for data collection per training step

        NetworkParams networkParams = getNetworkParams();
        NetworkParams gradient = networkParams.skeleton();

        for (int i = 0; i < trainingExamples.size(); i++) {
            System.err.println("Training Example " + "(" + i + ")");
            NetworkParams grad_i = backpropagation(trainingExamples.get(i), networkParams);

            //System.err.println("GRADIENT (" + i + "):\n" + grad_i);
            gradient = gradient.plus(grad_i); //sum gradients of each training example

            System.err.println("i = " + i + " -> GRADIENT:\n" + gradient);
            //allGradients.add(grad_i);
        }

        //Debug.printAll(allGradients, System.err);
        return gradient.divide(trainingExamples.size()); //take the average
    }

    private NetworkParams backpropagation(TrainingExample trainingExample, NetworkParams networkParams) {
        NetworkParams gradients = networkParams.skeleton();
        Tensor allActivations = predictWithAllStats(trainingExample.X); //PREDICTION

        System.err.println("PREDICTED FROM " + trainingExample.toString());

        //backpropagation
        int L = layers.length - 1;

        Layer currentLayer = layers[L];
        SimpleMatrix prevActivations_L = allActivations.get(L - 1);

        SimpleMatrix error = baseError(   // error_L
                allActivations.getLast(), // predicted activations
                trainingExample.Y,        // actual activations
                currentLayer.Z(prevActivations_L),
                currentLayer.getActualMechanics()
        );
        gradients.set(L,
                Operations.plotMatrix(prevActivations_L, error),   // TW
                error.copy()                                       // Tb
        );

        SimpleMatrix W_lplus1 = currentLayer.getWeights();

        for (int l = L - 1; l > 0; --l) {
            currentLayer = layers[l]; // switch to current layer
            SimpleMatrix prevActivations_l = allActivations.get(l - 1);
            SimpleMatrix primedActivations = currentLayer.primedActivations(prevActivations_l);

            error = (W_lplus1.transpose().mult(error)).elementMult(primedActivations); // propagate backwards
            SimpleMatrix gradJ$W_l = Operations.plotMatrix(prevActivations_l, error);

            gradients.set(l,
                    gradJ$W_l,   // TW
                    error.copy() // Tb: gradJ$b_l = error_l
            );

            W_lplus1 = currentLayer.getWeights();
        }

        return gradients;
    }

    //gradient of loss with respect to activations multiplied by primed activations
    private SimpleMatrix baseError(SimpleMatrix predictedActivations, SimpleMatrix actualActivations, SimpleMatrix z, List<Mechanics> mechsList) {
        double[] error = new double[mechsList.size()];
        double[] zs = Operations.toArray(z);
        for (int i = 0; i < error.length; i++) {
            Mechanics funcs = mechsList.get(i);

            double delJ_delAhat = funcs.loss.applyPartialDerivative(predictedActivations.get(i), actualActivations.get(i)); //actualActivations is fixed because this is with respect to predictedActivations
            double delA_delZ = funcs.activation.autoApplyPartialDerivative(zs, i);

            error[i] = delJ_delAhat * delA_delZ; // delJ_delZ
        }

        return Operations.colVector(error);
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
        System.err.println("Layer-wise activations: \n" + allActivations);
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
