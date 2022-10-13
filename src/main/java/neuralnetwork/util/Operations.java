package neuralnetwork.util;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
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
        Arrays.fill(arr, val);

        return arr;
    }

    public static SimpleMatrix colVector(double[] data) {
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }

    public static SimpleMatrix rowVector(double[] data) {
        return SimpleMatrix.wrap(new DMatrixRMaj(data)).transpose();
    }

    public static SimpleMatrix colVector(int fill, int length) {
        double[] data = new double[length];
        Arrays.fill(data, fill);

        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }

    public static SimpleMatrix rowVector(int fill, int length) {
        double[] data = new double[length];
        Arrays.fill(data, fill);

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

         return (f.applyAsDouble(x + h) - f.applyAsDouble(x)) / h;
    }

    public static DoubleUnaryOperator derivative(DoubleUnaryOperator f, int specificity) {
        double h = 1.0 / specificity;

        return (x) -> (f.applyAsDouble(x + h) - f.applyAsDouble(x)) / h;
    }

    public static double[] toArray(SimpleMatrix data) {
        double[] arr = new double[data.numRows() * data.numCols()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = data.get(i);
        }

        return arr;
    }

    //like a plot of land or area, or punnett square
    //input 2 vectors and out comes a matrix
    public static SimpleMatrix plotMatrix(SimpleMatrix horizontal, SimpleMatrix vertical) {
        int rows = Math.max(vertical.numRows(), vertical.numCols()), cols = Math.max(horizontal.numCols(), horizontal.numRows());
        double[][] data = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                data[row][col] = horizontal.get(col) * vertical.get(row);
            }
        }

        return matrix(data);
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

    public static String matrixToString(SimpleMatrix matrix) {
        StringBuilder sb = new StringBuilder();
        if (isVector(matrix)) {
            sb.append(vectorToString(matrix));
        } else {
            sb.append(matrix.toString());
        }

        return sb.append(" (").append(matrix.numRows()).append("x").append(matrix.numCols()).append(")")
                 .toString();
    }
}
