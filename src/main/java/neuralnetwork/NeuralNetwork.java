package neuralnetwork;

import math.Tensor;
import neuralnetwork.training.LearningAlgorithm;
import neuralnetwork.training.NetworkParams;
import neuralnetwork.training.TrainingExample;
import neuralnetwork.util.MechNetworkIndex;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.LinkedList;
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

    public NeuralNetwork setMechanics(MechNetworkIndex... mechs) {
        for (MechNetworkIndex mechI : mechs) {
            layers[mechI.layer].setMechanics(mechI.mechIndex);
        }

        return this;
    }

    //randomize w and b everywhere
    public void reset() {
        NetworkParams networkParams = getNetworkParams(); // get current params of ANN, specifically the "shape" matters here

        DoubleUnaryOperator randomization = (theta) -> { return Math.random(); }; // reset operation

        setNetworkParams(networkParams.applyEntrywise(randomization));
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
        for (int l = 0; l < layers.length; l++) {
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
        List<NetworkParams> allNetworkParams = new LinkedList<>(); //for data collection per training step

        NetworkParams networkParams = getNetworkParams();
        NetworkParams gradient = getNetworkParams().fill(0);

        for (TrainingExample trainingExample : trainingExamples) {
            NetworkParams gradients = backpropagation(trainingExample, networkParams);

            gradient = gradient.plus(gradients); //sum gradients of each training example
            allNetworkParams.add(gradients);
        }

        System.out.println(allNetworkParams);
        return gradient.divide(trainingExamples.size()); //take the average
    }

    private NetworkParams backpropagation(TrainingExample trainingExample, NetworkParams networkParams) {
        Tensor T_W = new Tensor(layers.length), T_b = new Tensor(layers.length);
        Tensor allActivations = predictWithAllStats(trainingExample.X);

        //fill input layer with 0 so no change occurs
        SimpleMatrix input_W = networkParams.TW.get(0).copy(), input_b = networkParams.Tb.get(0).copy();
        input_W.fill(0);
        input_b.fill(0);
        T_W.set(0, input_W);
        T_b.set(0, input_b);

        //backpropagation
        Layer currentLayer = layers[layers.length - 1];
        SimpleMatrix activations = allActivations.getLast(), prevActivations = allActivations.get(layers.length - 2);
        List<Mechanics> mechsList = currentLayer.getActualMechanics();

        SimpleMatrix error = baseError(activations, trainingExample.Y, currentLayer.Z(prevActivations), mechsList);
        for (int l = layers.length - 1; l > 0; l--) {
            SimpleMatrix W_lplus1 = currentLayer.getWeights();
            currentLayer = layers[l]; // switch to current layer

            SimpleMatrix primedActivations = currentLayer.primedActivations(allActivations.get(l));

            error = (W_lplus1.transpose().mult(error)).elementMult(primedActivations);
            SimpleMatrix gradJ$W_l = Operations.plotMatrix(allActivations.get(l - 1), error);

            T_W.set(l, gradJ$W_l);
            T_b.set(l, error); // gradJ$b_l = error_l
        }

        return new NetworkParams(T_W, T_b);
    }

    //gradient of loss with respect to activations multiplied by primed activations
    private SimpleMatrix baseError(SimpleMatrix predictedActivations, SimpleMatrix actualActivations, SimpleMatrix z, List<Mechanics> mechsList) {
        double[] error = new double[mechsList.size()];
        double[] zs = Operations.toArray(z);
        for (int i = 0; i < error.length; i++) {
            Mechanics mech = mechsList.get(i);

            double ahat = predictedActivations.get(i);
            double a = actualActivations.get(i);

            error[i] = mech.loss.applyPartialDerivative(ahat, a) * mech.activation.autoApplyPartialDerivative(zs, i);
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

        //set weights to 0
        SimpleMatrix w_0 = Operations.matrix(X.length, 1); //numCols = 1 because it can be pretty much anything, so smallest is better for less space
        w_0.zero();

        currentLayer.setWeights(w_0);

        //This seems redundant, but it's because of looking at the stats and data of the neural network later
        currentLayer.setBiases(X);

        //Forward Propagation
        SimpleMatrix activations = currentLayer.activations(w_0.copy());
        allActivations.set(0, activations.copy()); //data recording step

        for (int i = 1; i < layers.length; i++) {
            currentLayer = layers[i];
            activations = currentLayer.activations(activations); // a' = Activations(Wa + b)

            allActivations.set(i, activations.copy()); //data recording step
        }

        System.out.println("Layer-wise activations: \n" + allActivations);
        return allActivations;
    }

    public SimpleMatrix fastPredict(double[] X) { //doesn't record data
        Layer currentLayer = layers[0]; //current layer is input layer

        //set weights to 0
        SimpleMatrix w_0 = Operations.matrix(X.length, 1); //numCols = 1 because it can be pretty much anything, so smallest is better for less space
        w_0.zero();

        currentLayer.setWeights(w_0);

        //This seems redundant, but it's because of looking at the stats and data of the neural network later
        currentLayer.setBiases(X);

        //Forward Propagation
        SimpleMatrix activations = currentLayer.activations(w_0.copy());

        for (int i = 1; i < layers.length; i++) {
            currentLayer = layers[i];
            activations = currentLayer.activations(activations); // a' = Activations(Wa + b)
        }

        return activations;
    }
}
