package neuralnetwork;

import neuralnetwork.util.Mechanics;
import org.ejml.simple.SimpleMatrix;

public class InputNeuron extends Neuron {
    public InputNeuron(double a) {
        super(a);
    }

    @Override
    public double activation(SimpleMatrix prevActivations) { //param doesn't matter; it can be null
        return bias; //activation for neurons in the input layer is the bias
    }
}
