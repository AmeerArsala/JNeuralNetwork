package neuralnetwork.loss;

import neuralnetwork.Globals;
import neuralnetwork.util.Operations;

import java.util.function.DoubleUnaryOperator;

public enum Loss {
    SquaredError((yhat, y) -> {
        double error = yhat - y;
        return (error*error);
    }, (yhat, y) -> (2.0 * (yhat - y))),
    AbsoluteError((yhat, y) -> Math.abs(yhat - y)),
    BinaryCrossentropy((yhat, y) -> ((y * -Math.log(yhat)) + ((1 - y) * -Math.log(1 - yhat))) ), // logistic loss
    CategoricalCrossentropy((a_i, y_i) -> -Math.log(a_i)), //TODO: improve this
    None((yhat, y) -> 0);

    private final LossFunction lossFunc, partialDerivative_yhat;
    private Loss(LossFunction L) {
        lossFunc = L;
        partialDerivative_yhat = (yhat, y) -> Operations.derivative(lossFunc.fixedY(y), yhat, Globals.DERIVATIVE_SPECIFICITY);
    }

    private Loss(LossFunction L, LossFunction d_dyhat$L) {
        lossFunc = L;
        partialDerivative_yhat = d_dyhat$L;
    }

    public double apply(double yhat, double y) { //
        return lossFunc.LOSS(yhat, y);
    }

    public double applyPartialDerivative(double yhat, double y) {
        return partialDerivative_yhat.LOSS(yhat, y);
    }
}
