package neuralnetwork.training;

import java.util.ArrayList;
import java.util.List;

public abstract class LearningAlgorithm {
    protected final List<TrainingExample> trainingExamples = new ArrayList<>();

    public LearningAlgorithm() {}

    public LearningAlgorithm init(List<TrainingExample> allTrainingExamples) {
        trainingExamples.addAll(allTrainingExamples);
        return this;
    }

    public abstract List<TrainingExample> shuffleData();

    public abstract NetworkParams learnStep(NetworkParams currentParams, NetworkParams gradient);

    public abstract boolean doesConverge();

    //Learning Algorithm Presets

    //why do we need multiple epochs? gradient descent steps don't completely go towards the minimum, only taking smaller
    public static LearningAlgorithm BatchGradientDescent(double learningRate, int epochs) {
        return new LearningAlgorithm() {
            private int i = 0;

            @Override
            public List<TrainingExample> shuffleData() {
                return trainingExamples;
            }

            @Override
            public NetworkParams learnStep(NetworkParams currentParams, NetworkParams gradient) {
                NetworkParams next = currentParams.minus(gradient.scale(learningRate));

                ++i; //1 batch = 1 epoch in BGD
                return next;
            }

            @Override
            public boolean doesConverge() {
                return i >= epochs;
            }
        };
    }

    //TODO: add stochastic gradient descent, adam algorithm, and least squares
}
