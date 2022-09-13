package main;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.training.LearningAlgorithm;
import neuralnetwork.training.TrainingExample;
import neuralnetwork.util.Operations;
import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        System.out.println("Hi");

        int inputSize = 5;  //size of input layer
        int outputSize = 4; //size of output layer
        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{inputSize, 3, 5, 4, outputSize});

        /**
         * Inputs: [latitude, longitude, purchasing power / cost of living, level of technology, safety]
         * Outputs: [quality of life, food, people and their hospitality, fun]
         */
        double MAX_RATING = 10.0;
        List<TrainingExample> trainingExamples = Arrays.asList(
            new TrainingExample( // Japan
                    new double[]{36.204823, 138.252930, (103.12 / 83.33), 5.68, 87.105}, // input
                    vec(9.25, 9.0, 10.0, 8.0).divide(MAX_RATING)                   // output
            ),
            new TrainingExample( // USA
                    new double[]{36.778259, -119.417931, (99.88 / 69.92), 6.24, 51.59}, // input
                    vec(8.0, 8.5, 7.5, 8.0).divide(MAX_RATING)                    // output
            ),
            new TrainingExample( // Bosnia
                    new double[]{43.915886, 17.679075, (46.51 / 32.24), 3.3, 57.29},  // input
                    vec(7.5, 9.0, 10.0, 8.0).divide(MAX_RATING)                 // output
            ),
            new TrainingExample( // Mexico
                    new double[]{32.514946, -117.038246, (41.10 / 35.14), 4.13, 46.10}, // input
                    vec(5.5, 6.5, 6.5, 7.0).divide(MAX_RATING)                    // output
            ),
            new TrainingExample( // Morocco
                    new double[]{31.791702, -7.092620, (32.67 / 29.71), 3.3, 52.63}, // input
                    vec(6.5, 6.0, 7.0, 6.0).divide(MAX_RATING)                 // output
            ),
            new TrainingExample( // Germany
                    new double[]{51.165691, 10.451526, (102.75 / 59.62), 5.08, 64.42}, // input
                    vec(7.5, 7.0, 3.0, 2.5).divide(MAX_RATING)                   // output
            ),
            new TrainingExample( // Afghanistan
                    new double[]{33.93911, 67.709953, (5.44 / 23.26), 2.9, 0.0}, // input
                    vec(0.0, 4.0, 0.0, 0.0).divide(MAX_RATING)                   // output
            )
        );

        double learningRate = 0.01;
        neuralNetwork.train(trainingExamples, LearningAlgorithm.BatchGradientDescent(learningRate, 30));

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("Enter input (type STOP to terminate program)");
            String userInput = scanner.nextLine();

            if (userInput.equalsIgnoreCase("STOP")) {
                break;
            }

            String[] inputs = userInput.split(", ");

            double[] X = new double[inputs.length];
            for (int i = 0; i < X.length; i++) {
                X[i] = Double.parseDouble(inputs[i]);
            }

            SimpleMatrix output = neuralNetwork.predict(X);

            System.out.println("OUTPUT: " + Operations.vectorToString(output));
        }
    }

    private static SimpleMatrix vec(double... data) { //column vector
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }
}
