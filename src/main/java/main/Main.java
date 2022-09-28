package main;

import com.opencsv.CSVReader;
import debug.Debug;
import example.tool.CountryCodes;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.activation.Activation;
import neuralnetwork.loss.Loss;
import neuralnetwork.training.LearningAlgorithm;
import neuralnetwork.training.TrainingExample;
import neuralnetwork.util.MechIndex;
import neuralnetwork.util.MechNetworkIndex;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        System.out.println("Hello World!");

        worldPrediction();
    }
    private static void worldPrediction() {
        int inputSize = 5;  //size of input layer
        int outputSize = 4; //size of output layer
        NeuralNetwork neuralNetwork = new NeuralNetwork(new int[]{inputSize, 3, 5, 4, outputSize});

        neuralNetwork.setDenseMechanics(
                new MechIndex(1, new Mechanics(Activation.Sigmoid, Loss.SquaredError)),
                new MechIndex(2, new Mechanics(Activation.Sigmoid, Loss.SquaredError)),
                new MechIndex(3, new Mechanics(Activation.Sigmoid, Loss.SquaredError)),
                new MechIndex(4, new Mechanics(Activation.Sigmoid, Loss.SquaredError))
        );

        final double MAX_RATING = 10.0;
        List<TrainingExample> trainingExamples = worldAreaRatingTrainingData(MAX_RATING);

        double learningRate = 0.01;
        LearningAlgorithm learningAlgorithm = LearningAlgorithm.BatchGradientDescent(learningRate, 30);

        neuralNetwork.train(trainingExamples, learningAlgorithm);

        Scanner scanner = new Scanner(System.in);
        final String formatTutorialText = "Format for input is: 'latitude, longitude, purchasing power / cost of living, level of technology, safety'\nOutput is formatted as: 'quality of life, food, people and their hospitality, fun'";
        while (true) {
            System.out.println(formatTutorialText + "\nEnter input (Either type the name or code of a country or use the format described above. Type STOP to terminate program): ");
            String userInput = scanner.nextLine();

            if (userInput.equalsIgnoreCase("STOP")) {
                break;
            }

            double[] X = new double[inputSize]; //inputs
            String areaName;
            if (Character.isAlphabetic(userInput.charAt(0))) { //name or code of a country was typed
                areaName = userInput;

                //labeled columns
                final int
                        COUNTRY_CODE = 0,
                        LATITUDE = 1,
                        LONGITUDE = 2,
                        COUNTRY_NAME = 3,
                        QOL = 4,
                        COST_OF_LIVING = 5,
                        PURCHASING_POWER = 6,
                        SAFETY = 7;

                int countryID;
                if (userInput.length() == 2) { //country code typed
                    countryID = COUNTRY_CODE;
                } else { //name of country typed
                    countryID = COUNTRY_NAME;
                }

                try {
                    FileReader fileReader = new FileReader("res/data/country_data.csv");

                    CSVReader csvReader = new CSVReader(fileReader);
                    String[] nextRow;

                    while ((nextRow = csvReader.readNext()) != null) {
                        Debug.printAll(nextRow, System.err);
                        if (nextRow[countryID].equals(userInput)) {
                            //TODO: do this in a better way
                            X[0] = Double.parseDouble(nextRow[LATITUDE]);
                            X[1] = Double.parseDouble(nextRow[LONGITUDE]);

                            double purchasingPower = Double.parseDouble(nextRow[PURCHASING_POWER]);
                            double costOfLiving = Double.parseDouble(nextRow[COST_OF_LIVING]);

                            X[2] = purchasingPower == 0.0 ? 0.0 : purchasingPower / costOfLiving;
                            X[3] = Double.parseDouble(nextRow[QOL]);
                            X[4] = Double.parseDouble(nextRow[SAFETY]);

                            break;
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else {
                areaName = "this area";
                String[] inputs = userInput.split(", ");

                for (int i = 0; i < X.length; i++) {
                    X[i] = Double.parseDouble(inputs[i]);
                }
            }

            SimpleMatrix output = neuralNetwork.predict(X);

            System.out.println("OUTPUT: " + Operations.vectorToString(output) + "\n");
            System.out.println("On a scale of 1-10, " + areaName + " has:\n" +
                    "Quality of Life: " + (output.get(0) * MAX_RATING) + "\n" +
                    "Food: " + (output.get(1) * MAX_RATING) + "\n" +
                    "People and their Hospitality: " + (output.get(2) * MAX_RATING) + "\n" +
                    "Fun: " + (output.get(3) * MAX_RATING)
            );
        }
    }

    static List<TrainingExample> worldAreaRatingTrainingData(double MAX_RATING) {
        /**
         * Inputs: [latitude, longitude, purchasing power / cost of living, level of technology, safety]
         * Outputs: [quality of life, food, people and their hospitality, fun]
         */
        return Arrays.asList(
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
                        new double[]{33.93911, 67.709953, (5.44 / 23.26), 2.9, 0.0},       // input
                        vec(0.0, 4.0, 0.0, 0.0).divide(MAX_RATING)                   // output
                )
        );
    }
    private static SimpleMatrix vec(double... data) { //column vector
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }
}
