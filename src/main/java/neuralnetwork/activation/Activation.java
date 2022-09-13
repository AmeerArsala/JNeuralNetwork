package neuralnetwork.activation;

import neuralnetwork.Globals;
import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.function.DoubleUnaryOperator;

public class Activation {
    public static final Activation Sigmoid = new Activation((z) -> (1.0 / (1.0 + Math.exp(-z))), (z) -> {
        double exp = Math.exp(-z);
        double ogdenom = 1 + exp;

        return exp / (ogdenom*ogdenom);
    });

    public static final Activation ReLU = new Activation((z) -> Math.max(0, z));

    public static final Activation tanh = new Activation((z) -> ((Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z))) );
    public static final Activation Softmax = new Activation((zs, i) -> (Math.exp(zs[i]) / Operations.sum(zs)),
                                                            (zs, i) -> Operations.derivative((zs_i) -> (Math.exp(zs_i) / Operations.sum(zs)), zs[i], Globals.DERIVATIVE_SPECIFICITY)); //TODO: change later
    public static final Activation Linear = new Activation((z) -> z);

    public static interface MultiInputActivator {
        public double applyAsDouble(double[] inputs, int i);
    }
    private DoubleUnaryOperator activationFunc, partialDerivative_z;
    private MultiInputActivator multiInputActivationFunc, multiInputPartialDerivative;
    public Activation(DoubleUnaryOperator func) {
        activationFunc = func;
        partialDerivative_z = (z) -> Operations.derivative(activationFunc, z, Globals.DERIVATIVE_SPECIFICITY);
    }

    public Activation(DoubleUnaryOperator func, DoubleUnaryOperator d_dz) {
        activationFunc = func;
        partialDerivative_z = d_dz;
    }

    public Activation(MultiInputActivator func, MultiInputActivator d_dz) {
        multiInputActivationFunc = func;
        multiInputPartialDerivative = d_dz;
    }

    public boolean isMultiInput() {
        return multiInputActivationFunc != null;
    }

    public double apply(double z) {
        return activationFunc.applyAsDouble(z);
    }

    public double applyPartialDerivative(double z) {
        return partialDerivative_z.applyAsDouble(z);
    }

    public SimpleMatrix apply(SimpleMatrix input) {
        SimpleMatrix output = new SimpleMatrix(input);
        for (int i = 0; i < output.numRows(); ++i) {
            for (int j = 0; j < output.numCols(); ++j) {
                output.set(i, j, activationFunc.applyAsDouble(output.get(i, j)));
            }
        }
        return output;
    }

    public double apply(double[] zs, int i) {
        return multiInputActivationFunc.applyAsDouble(zs, i);
    }

    public double applyPartialDerivative(double[] zs, int i) {
        return multiInputPartialDerivative.applyAsDouble(zs, i);
    }

    public double autoApply(double[] zs, int i) {
        if (activationFunc != null) {
            return activationFunc.applyAsDouble(zs[i]);
        } else {
            return multiInputActivationFunc.applyAsDouble(zs, i);
        }
    }

    public double autoApplyPartialDerivative(double[] zs, int i) {
        if (partialDerivative_z != null) {
            return partialDerivative_z.applyAsDouble(zs[i]);
        } else {
            return multiInputPartialDerivative.applyAsDouble(zs, i);
        }
    }
}
