package neuralnetwork;

import neuralnetwork.activation.Activation;
import neuralnetwork.loss.Loss;
import neuralnetwork.util.MechIndex;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Layer {
    private final List<Neuron> neurons;
    private Mechanics mechanics; //default activation and loss functions; can be overrided by neurons individually
    private final int prevLayerSize;

    public Layer(int size, int prevLayerSize) {
        this(size, new Mechanics(Activation.Linear, Loss.None), prevLayerSize);
    }

    public Layer(int size, Mechanics mechanics, int prevLayerSize) {
        this.mechanics = mechanics;
        this.prevLayerSize = prevLayerSize;
        neurons = constructNeurons(size, mechanics);
        initializeWeightsAndBiases();
    }

    public Layer(int size, Mechanics[] mechanics, int prevLayerSize) {
        neurons = constructNeurons(size, mechanics);
        this.mechanics = new Mechanics(Activation.Linear, Loss.None);
        this.prevLayerSize = prevLayerSize;
        initializeWeightsAndBiases();
    }

    private void initializeWeightsAndBiases() {
        for (Neuron neuron : neurons) {
            double[] data = Operations.fill(0.0, prevLayerSize);
            neuron.initWandB(Operations.rowVector(data), 0.0);
        }
    }

    public int size() {
        return neurons.size();
    }

    public Layer setMechanics(MechIndex... mechs) {
        for (MechIndex mechI : mechs) {
            neurons.get(mechI.i).mechanics = mechI.mechanics;
        }

        return this;
    }

    public void setDenseMechanics(Mechanics mechs) {
        for (Neuron neuron : neurons) {
            neuron.mechanics = mechs;
        }

        mechanics = mechs;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public Neuron get(int i) {
        return neurons.get(i);
    }

    public Mechanics getStandardMechanics() {
        return mechanics;
    }

    public List<Mechanics> getActualMechanics() {
        List<Mechanics> mechs = new LinkedList<>();
        for (Neuron neuron : neurons) {
            mechs.add(neuron.mechanics);
        }

        return mechs;
    }

    public SimpleMatrix getWeights() { // weight matrix W
        double[][] W = new double[neurons.size()][prevLayerSize]; //cols = prevLayerSize

        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[i].length; j++) {
                W[i][j] = neurons.get(i).weights.get(j); //weights is a vector so it's ok
            }
        }

        return Operations.matrix(W);
    }

    public SimpleMatrix getBiases() {
        double[] b = new double[neurons.size()];
        for (int i = 0; i < b.length; i++) {
            b[i] = neurons.get(i).bias;
        }

        return Operations.colVector(b);
    }

    public void setBiases(double[] biases) {
        for (int i = 0; i < biases.length; i++) {
            neurons.get(i).bias = biases[i];
        }
    }

    public void setWeights(SimpleMatrix W) {
        for (int i = 0, rows = W.numRows(); i < rows; i++) {
            neurons.get(i).weights = W.extractVector(true, i);
        }
    }

    public void zeroWeights() {
        for (Neuron neuron : neurons) {
            neuron.weights.zero();
        }
    }

    public void zeroBiases() {
        for (Neuron neuron : neurons) {
            neuron.bias = 0.0;
        }
    }

    public SimpleMatrix Z(SimpleMatrix prevActivations) {
        // z = Wa + b
        // a is prevActivations column vector
        // b is biases column vector
        // z is column vector

        SimpleMatrix W = getWeights();
        SimpleMatrix b = getBiases();

        return W.mult(prevActivations).plus(b);
    }

    public SimpleMatrix activations(SimpleMatrix prevActivations) { //column vector of activations
        // a' = Activation(Wa + b)
        // a is prevActivations column vector
        // b is biases column vector
        // a' is new activations column vector

        SimpleMatrix W = getWeights();
        SimpleMatrix b = getBiases();

        SimpleMatrix a = W.mult(prevActivations).plus(b); // z

        for (int i = 0, length = neurons.size(); i < length; ++i) {
            a.set(i, neurons.get(i).mechanics.activation.apply(a.get(i)));
        }

        return a;
    }

    //activations with respect to respective z
    public SimpleMatrix primedActivations(SimpleMatrix prevActivations) {
        SimpleMatrix W = getWeights();
        SimpleMatrix b = getBiases();

        SimpleMatrix a = W.mult(prevActivations).plus(b); // z
        double[] zs = Operations.toArray(a);

        for (int i = 0, length = neurons.size(); i < length; ++i) {
            a.set(i, neurons.get(i).mechanics.activation.autoApplyPartialDerivative(zs, i));
        }

        return a;
    }

    public static List<Neuron> constructNeurons(int size, Mechanics mech) {
        List<Neuron> ns = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            ns.add(new Neuron(mech));
        }

        return ns;
    }

    public static List<Neuron> constructNeurons(int size, Mechanics[] mechs) {
        List<Neuron> ns = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            ns.add(new Neuron(mechs[i]));
        }

        return ns;
    }
}
