package neuralnetwork.util;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.util.function.DoubleUnaryOperator;

public class Operations {

    public static double sum(double[] nums) {
        double sum = 0.0;
        for (int i = 0; i < nums.length; ++i) {
            sum += nums[i];
        }

        return sum;
    }

    public static double[] fill(double val, int length) {
        double[] arr = new double[length];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = val;
        }

        return arr;
    }

    public static SimpleMatrix colVector(double[] data) {
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }

    public static SimpleMatrix rowVector(double[] data) {
        return SimpleMatrix.wrap(new DMatrixRMaj(data)).transpose();
    }

    public static SimpleMatrix matrix(double[][] data) {
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }

    public static SimpleMatrix matrix(int numRows, int numCols) {
        return SimpleMatrix.wrap(new DMatrixRMaj(numRows, numCols));
    }

    public static SimpleMatrix map(SimpleMatrix input, DoubleUnaryOperator func) {
        SimpleMatrix output = new SimpleMatrix(input);
        for (int i = 0; i < output.numRows(); ++i) {
            for (int j = 0; j < output.numCols(); ++j) {
                output.set(i, j, func.applyAsDouble(output.get(i, j)));
            }
        }

        return output;
    }

    public static SimpleMatrix mapFromVector(SimpleMatrix vec, DoubleUnaryOperator func) {
        SimpleMatrix output = new SimpleMatrix(vec);
        for (int i = 0; i < output.numRows(); ++i) {
            output.set(i, func.applyAsDouble(output.get(i)));
        }

        return output;
    }

    public static double derivative(DoubleUnaryOperator f, double x, int specificity) {
         double h = 1.0 / specificity;

         return f.applyAsDouble(x + h) / h;
    }

    public static DoubleUnaryOperator derivative(DoubleUnaryOperator f, int specificity) {
        double h = 1.0 / specificity;

        return (x) -> f.applyAsDouble(x + h) / h;
    }

    public static double[] toArray(SimpleMatrix data) {
        double[] arr = new double[data.numRows() * data.numCols()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = data.get(i);
        }

        return arr;
    }

    //like a plot of land or area
    public static SimpleMatrix plotMatrix(SimpleMatrix horizontal, SimpleMatrix vertical) {
        SimpleMatrix output = matrix(horizontal.numCols(), vertical.numRows());
        for (int col = 0; col < horizontal.numCols(); col++) {
            for (int row = 0; row < vertical.numRows(); row++) {
                output.set(row, col, horizontal.get(col) * vertical.get(row));
            }
        }

        return output;
    }

    public static boolean isVector(SimpleMatrix matrix) {
        return matrix.numRows() == 1 || matrix.numCols() == 1;
    }

    public static String vectorToString(SimpleMatrix vec) {
        StringBuilder sb = new StringBuilder("<").append(vec.get(0));
        for (int j = 1, length = vec.numRows() * vec.numCols(); j < length; j++) {
            sb.append(", ").append(vec.get(j));
        }
        sb.append(">");

        return sb.toString();
    }
}
