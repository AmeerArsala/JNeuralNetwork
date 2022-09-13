package math;

import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

public class Tensor { //very basic tensor
    private SimpleMatrix[] matrices;

    public Tensor(int length) {
        matrices = new SimpleMatrix[length];
    }

    public Tensor(SimpleMatrix[] matrices) {
        this.matrices = matrices;
    }

    public int size() {
        return matrices.length;
    }

    public int size(int i) {
        SimpleMatrix matrix = matrices[i];

        return matrix.numRows() * matrix.numCols();
    }

    public SimpleMatrix get(int i) {
        return matrices[i];
    }

    public SimpleMatrix getLast() {
        return matrices[matrices.length - 1];
    }

    public void set(int i, SimpleMatrix matrix) {
        matrices[i] = matrix;
    }

    public void set(Tensor tensor) {
        matrices = tensor.matrices;
    }

    public Tensor plus(Tensor other) {
        if (other.matrices.length != matrices.length) {
            throw new IndexOutOfBoundsException();
        }

        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].plus(other.matrices[i]);
        }

        return new Tensor(mats);
    }

    public Tensor plus(SimpleMatrix M) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].plus(M);
        }

        return new Tensor(mats);
    }

    public Tensor plus(double val) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].plus(val);
        }

        return new Tensor(mats);
    }

    public Tensor minus(Tensor other) {
        if (other.matrices.length != matrices.length) {
            throw new IndexOutOfBoundsException();
        }

        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].minus(other.matrices[i]);
        }

        return new Tensor(mats);
    }

    public Tensor minus(SimpleMatrix M) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].minus(M);
        }

        return new Tensor(mats);
    }

    public Tensor minus(double val) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].minus(val);
        }

        return new Tensor(mats);
    }

    public Tensor elementMult(SimpleMatrix M) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].mult(M);
        }

        return new Tensor(mats);
    }

    public Tensor entrywiseMult(SimpleMatrix M) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].elementMult(M);
        }

        return new Tensor(mats);
    }

    public Tensor scale(double scalar) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].scale(scalar);
        }

        return new Tensor(mats);
    }

    public Tensor divide(double val) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrices[i].divide(val);
        }

        return new Tensor(mats);
    }

    public Tensor apply(UnaryOperator<SimpleMatrix> operation) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = operation.apply(matrices[i]);
        }

        return new Tensor(mats);
    }

    public Tensor applyEntrywise(DoubleUnaryOperator operation) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = Operations.map(matrices[i], operation);
        }

        return new Tensor(mats);
    }

    public Tensor fill(SimpleMatrix matrix) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = matrix.copy();
        }

        return new Tensor(mats);
    }

    public Tensor fill(double val) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            SimpleMatrix matrix = matrices[i].copy();
            matrix.fill(val);
            mats[i] = matrix;
        }

        return new Tensor(mats);
    }

    public Tensor fill(int i, double val) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];

        SimpleMatrix matrix = matrices[i].copy();
        matrix.fill(val);
        mats[i] = matrix;

        for (int j = 1; j < mats.length; ++j) {
            mats[i] = matrices[j].copy();
        }

        return new Tensor(mats);
    }

    public Tensor defineMatrixShape(int rows, int cols) {
        SimpleMatrix[] mats = new SimpleMatrix[matrices.length];
        for (int i = 0; i < mats.length; ++i) {
            mats[i] = Operations.matrix(rows, cols);
        }

        return new Tensor(mats);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < matrices.length; i++) {
            sb.append("[").append(i).append("]: ");
            SimpleMatrix matrix = matrices[i];
            if (Operations.isVector(matrix)) {
                sb.append("<").append(matrix.get(0));
                for (int j = 1, length = matrix.numRows() * matrix.numCols(); j < length; j++) {
                    sb.append(", ").append(matrix.get(j));
                }
                sb.append(">");
            } else {
                sb.append(matrix.toString());
            }
        }

        return sb.toString();
    }
}
