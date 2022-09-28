package neuralnetwork.training;

import math.Tensor;
import org.ejml.simple.SimpleMatrix;

import java.util.function.DoubleUnaryOperator;

public class NetworkParams {
    public final Tensor TW, Tb;

    public NetworkParams(Tensor TW, Tensor Tb) {
        this.TW = TW;
        this.Tb = Tb;
    }

    public NetworkParams(int layers) {
        TW = new Tensor(layers);
        Tb = new Tensor(layers);
    }

    public NetworkParams plus(NetworkParams other) {
        Tensor TW_2 = TW.plus(other.TW);
        Tensor Tb_2 = Tb.plus(other.Tb);

        return new NetworkParams(TW_2, Tb_2);
    }

    public NetworkParams minus(NetworkParams other) {
        Tensor TW_2 = TW.minus(other.TW);
        Tensor Tb_2 = Tb.minus(other.Tb);

        return new NetworkParams(TW_2, Tb_2);
    }

    public NetworkParams scale(double scalar) {
        return new NetworkParams(TW.scale(scalar), Tb.scale(scalar));
    }

    public NetworkParams divide(double val) {
        return new NetworkParams(TW.divide(val), Tb.divide(val));
    }

    public NetworkParams fill(double val) {
        return new NetworkParams(TW.fill(val), Tb.fill(val));
    }

    public NetworkParams applyEntrywise(DoubleUnaryOperator operation)  {
        Tensor TW_2 = TW.applyEntrywise(operation);
        Tensor Tb_2 = Tb.applyEntrywise(operation);

        return new NetworkParams(TW_2, Tb_2);
    }

    public void set(NetworkParams np) {
        TW.set(np.TW);
        Tb.set(np.Tb);
    }

    public void set(int i, SimpleMatrix W, SimpleMatrix b) {
        TW.set(i, W);
        Tb.set(i, b);
    }

    public NetworkParams skeleton() {
        return fill(0);
    }

    @Override
    public String toString() {
        return "NETWORK PARAMS {\nTensor W:\n" + TW.toString() + "\nTensor b:\n" + Tb.toString() + "}\n";
    }
}
