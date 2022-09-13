package neuralnetwork;

import neuralnetwork.activation.Activation;
import neuralnetwork.loss.Loss;
import neuralnetwork.util.Mechanics;
import org.ejml.simple.SimpleMatrix;

public class Neuron {
    protected SimpleMatrix weights;
    protected double bias;
    protected Mechanics mechanics;

    public Neuron(Mechanics mechanics) {
        this.mechanics = mechanics;
    }

    protected Neuron(SimpleMatrix w, double b, Mechanics mech) {
        weights = w;
        bias = b;
        mechanics = mech;
    }

    protected Neuron(double b) {
        bias = b;
    }

    protected void initWandB(SimpleMatrix w, double b) {
        weights = w;
        bias = b;
    }

    public SimpleMatrix getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public Activation getActivationFunction() {
        return mechanics.activation;
    }

    public Loss getLossFunction() {
        return mechanics.loss;
    }

    public double z(SimpleMatrix prevActivations) {
        return weights.dot(prevActivations) + bias;
    }

    public double activation(SimpleMatrix prevActivations) {
        double z = weights.dot(prevActivations) + bias;
        return mechanics.activation.apply(z);
    }
}
