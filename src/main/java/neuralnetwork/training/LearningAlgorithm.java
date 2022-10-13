package neuralnetwork.training;

import math.Tensor;
import neuralnetwork.Layer;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

public abstract class LearningAlgorithm {
    protected final List<TrainingExample> trainingExamples = new ArrayList<>();
    protected NetworkParams skel;

    public LearningAlgorithm() {}

    public LearningAlgorithm init(List<TrainingExample> allTrainingExamples, NetworkParams skeleton) {
        trainingExamples.addAll(allTrainingExamples);
        skel = skeleton;
        return this;
    }

    public NetworkParams learnStep(NeuralNetwork network, NetworkParams currentParams) {
        return learnStep(network, currentParams, shuffleData());
    }

    public abstract List<TrainingExample> shuffleData();

    protected abstract NetworkParams learnStep(NeuralNetwork network, NetworkParams currentParams, List<TrainingExample> trainingExamples);

    public abstract boolean doesConverge();

    public NetworkParams calculateGradient(NeuralNetwork neuralNetwork, List<TrainingExample> trainingExamples) {
        NetworkParams gradient = skel.skeleton();

        for (int i = 0; i < trainingExamples.size(); i++) {
            NetworkParams grad_i = backpropagation(trainingExamples.get(i), neuralNetwork);

            //System.err.println("GRADIENT (" + i + "):\n" + grad_i);
            gradient = gradient.plus(grad_i); //sum gradients of each training example

            //System.err.println("Training Example i = " + i + " -> GRADIENT:\n" + gradient);
        }

        gradient = gradient.divide(trainingExamples.size()); //take the average

        System.err.println("FINAL GRADIENT: " + gradient);
        return gradient;
    }

    private NetworkParams backpropagation(TrainingExample trainingExample, NeuralNetwork neuralNetwork) {
        NetworkParams gradients = skel.skeleton();
        Tensor allActivations = neuralNetwork.predictWithAllStats(trainingExample.X); //PREDICTION

        //System.err.println("PREDICTED FROM " + trainingExample.toString());

        //backpropagation
        int L = neuralNetwork.getNumLayers() - 1;

        Layer currentLayer = neuralNetwork.getLayer(L);
        SimpleMatrix prevActivations_L = allActivations.get(L - 1);

        SimpleMatrix error = baseError(   // error_L
                allActivations.getLast(), // predicted activations
                trainingExample.Y,        // actual activations
                currentLayer.Z(prevActivations_L),
                currentLayer.getActualMechanics()
        );
        gradients.set(L,
                Operations.plotMatrix(prevActivations_L, error),   // TW
                error.copy()                                       // Tb
        );

        SimpleMatrix W_lplus1 = currentLayer.getWeights();

        for (int l = L - 1; l > 0; --l) {
            currentLayer = neuralNetwork.getLayer(l); // switch to current layer
            SimpleMatrix prevActivations_l = allActivations.get(l - 1);
            SimpleMatrix activationsPrime = currentLayer.activationsPrime(prevActivations_l);

            error = (W_lplus1.transpose().mult(error)).elementMult(activationsPrime); // propagate backwards
            SimpleMatrix gradJ$W_l = Operations.plotMatrix(prevActivations_l, error);

            gradients.set(l,
                    gradJ$W_l,   // TW
                    error.copy() // Tb: gradJ$b_l = error_l
            );

            W_lplus1 = currentLayer.getWeights();
        }

        return gradients;
    }

    //gradient of loss with respect to activations multiplied by primed activations
    public static SimpleMatrix baseError(SimpleMatrix predictedActivations, SimpleMatrix actualActivations, SimpleMatrix z, List<Mechanics> mechsList) {
        double[] error = new double[mechsList.size()];
        double[] zs = Operations.toArray(z);
        for (int i = 0; i < error.length; i++) {
            Mechanics funcs = mechsList.get(i);

            double delJ_delAhat = funcs.loss.applyPartialDerivative(predictedActivations.get(i), actualActivations.get(i)); //actualActivations is fixed because this is with respect to predictedActivations
            double delA_delZ = funcs.activation.autoApplyPartialDerivative(zs, i);

            error[i] = delJ_delAhat * delA_delZ; // delJ_delZ
        }

        return Operations.colVector(error);
    }

    //Learning Algorithm Presets
    public static LearningAlgorithm BatchGradientDescent(double learningRate, int epochs) { //why do we need multiple epochs? gradient descent steps don't completely go towards the minimum, only taking smaller
        return new LearningAlgorithm() {
            private int i = 0;

            @Override
            public List<TrainingExample> shuffleData() {
                return new ArrayList<>(trainingExamples);
            }

            @Override
            protected NetworkParams learnStep(NeuralNetwork network, NetworkParams currentParams, List<TrainingExample> data) {
                NetworkParams gradient = calculateGradient(network, data);
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

    public static LearningAlgorithm BatchGradientDescent(double learningRate, double convergenceThreshold) { //keeps going until convergence
        return new LearningAlgorithm() {
            private NetworkParams gradient;
            private double total;

            @Override
            public List<TrainingExample> shuffleData() {
                return new ArrayList<>(trainingExamples);
            }

            @Override
            protected NetworkParams learnStep(NeuralNetwork network, NetworkParams currentParams, List<TrainingExample> data) {
                gradient = calculateGradient(network, data);
                return currentParams.minus(gradient.scale(learningRate));
            }

            private void addAbs(double theta) { total += Math.abs(theta); }

            @Override
            public boolean doesConverge() {
                if (gradient == null) {
                    return false;
                }

                total = 0.0;
                gradient.forEach(this::addAbs);

                return total <= convergenceThreshold;
            }
        };
    }

    /*public static LearningAlgorithm OrdinaryLeastSquaresNormalEquation() {
        return new LearningAlgorithm() {
            private boolean learned = false;
            @Override
            public List<TrainingExample> shuffleData() {
                return null;
            }

            @Override
            protected NetworkParams learnStep(NeuralNetwork network, NetworkParams currentParams, List<TrainingExample> trainingExamples) {
                int rows = trainingExamples.size(), cols = currentParams.countParams();
                int outputs = currentParams.Tb.getLast().getNumElements();

                SimpleMatrix A = Operations.matrix(rows, cols);
                SimpleMatrix[] Y = new SimpleMatrix[outputs];

                for (int o = 0; o < outputs; o++) {
                    Y[o] = Operations.matrix(rows, 1);
                }

                for (int r = 0; r < rows; r++) {
                    TrainingExample trainingExample = trainingExamples.get(r);
                    for (int c = 0; c < cols; c++) {
                        A.set(r, c, trainingExample.X[c]);
                    }

                    for (int o = 0; o < outputs; o++) { // y.set(r, trainingExample.Y.get()); but for all o
                        Y[o].set(r, trainingExample.Y.get(o));
                    }
                }

                // Normal Equation
                SimpleMatrix x = A.transpose().mult(A).invert().mult(A.transpose()).mult(Y[0]);
                for (int o = 1; o < outputs; o++) {
                    SimpleMatrix xo = A.transpose().mult(A).invert().mult(A.transpose()).mult(Y[o]);
                    x = x.plus(xo);
                }

                x = x.divide(outputs);

                learned = true;
            }

            @Override
            public boolean doesConverge() {
                return learned;
            }
        }
    }*/

    //TODO: add stochastic gradient descent, adam algorithm, and least squares
}
