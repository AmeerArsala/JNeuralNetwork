package neuralnetwork.loss;

import java.util.function.DoubleUnaryOperator;

@FunctionalInterface
public interface LossFunction {
    public double LOSS(double yhat, double y); // yhat = prediction, y = target/actual

    public default DoubleUnaryOperator fixedY(double y) {
        return (double yhat) -> LOSS(yhat, y);
    }
}
