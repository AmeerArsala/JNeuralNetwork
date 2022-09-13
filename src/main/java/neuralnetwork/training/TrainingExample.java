package neuralnetwork.training;

import org.ejml.simple.SimpleMatrix;

public class TrainingExample {
    public final double[] X;
    public final SimpleMatrix Y;

    public TrainingExample(double[] X, SimpleMatrix Y) {
        this.X = X;
        this.Y = Y;
    }
}
