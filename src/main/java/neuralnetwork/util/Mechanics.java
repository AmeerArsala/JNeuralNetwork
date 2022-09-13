package neuralnetwork.util;

import neuralnetwork.activation.Activation;
import neuralnetwork.loss.Loss;

public class Mechanics {
    public final Activation activation;
    public final Loss loss;

    public Mechanics(Activation activation, Loss loss) {
        this.activation = activation;
        this.loss = loss;
    }
}
