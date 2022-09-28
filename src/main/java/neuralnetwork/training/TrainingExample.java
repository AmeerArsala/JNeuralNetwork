package neuralnetwork.training;

import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

public class TrainingExample {
    public final double[] X;
    public final SimpleMatrix Y;

    public TrainingExample(double[] X, SimpleMatrix Y) {
        this.X = X;
        this.Y = Y;
    }

    @Override
    public String toString() {
        return "TrainingExample {\nX: " + Arrays.toString(X) + "\nY: " + Operations.matrixToString(Y) + "\n}";
    }
}
