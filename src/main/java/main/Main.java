package main;

import example.CountryEvaluation;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.activation.Activation;
import neuralnetwork.loss.Loss;
import neuralnetwork.training.LearningAlgorithm;
import neuralnetwork.training.TrainingExample;
import neuralnetwork.util.MechIndex;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        worldPrediction();
    }
    private static void worldPrediction() {
        /**
         * Inputs: [latitude, longitude, purchasing power / cost of living, general quality of life, safety]
         * Outputs: [subjective quality of life, food, people and their hospitality, fun]
         */
        CountryEvaluation countryEvaluation = new CountryEvaluation();

        countryEvaluation.neuralNetwork.setDenseMechanics(
                new MechIndex(1, new Mechanics(Activation.Sigmoid, Loss.BinaryCrossentropy)),
                new MechIndex(2, new Mechanics(Activation.Sigmoid, Loss.BinaryCrossentropy))
        );

        double learningRate = 0.05;
        LearningAlgorithm learningAlgorithm = LearningAlgorithm.BatchGradientDescent(learningRate, 100);

        List<TrainingExample> trainingExamples = new ArrayList<>(countryEvaluation.sampleData());
        //trainingExamples.addAll(countryEvaluation.randomizedTrainingData(75, 0.0));

        countryEvaluation.neuralNetwork.train(trainingExamples, learningAlgorithm);
        countryEvaluation.run();
    }

    private static SimpleMatrix vec(double... data) { //column vector
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }
}
