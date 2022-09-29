package example;

import com.opencsv.CSVReader;
import example.tool.CountryColumns;
import misc.GeneralUtils;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.activation.Activation;
import neuralnetwork.loss.Loss;
import neuralnetwork.training.TrainingExample;
import neuralnetwork.util.MechIndex;
import neuralnetwork.util.Mechanics;
import neuralnetwork.util.Operations;
import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * Inputs: [latitude, longitude, purchasing power / cost of living, general quality of life, safety]
 * Outputs: [subjective quality of life, food, people and their hospitality, fun]
 */
public class CountryEvaluation {
    private static final int inputSize = 5;  //size of input layer
    private static final int outputSize = 4; //size of output layer

    public static final int Latitude = 0, Longitude = 1, AffordabilityRatio = 2, GeneralQoL = 3, Safety = 4; //input columns
    public static final double MAX_RATING = 10.0; //max output rating

    public static final CountryColumns csvCountry = new CountryColumns(
            0,
            3,
            1,
            2,
            4,
            5,
            6,
            7
    );

    private final List<String[]> countriesCSVdata;
    public final SimpleMatrix latitudeColumn, longitudeColumn, purchasingPowerColumn, costOfLivingColumn, affordabilityRatioColumn, qolColumn, safetyColumn;
    public final NeuralNetwork neuralNetwork;

    public CountryEvaluation() {
        neuralNetwork = new NeuralNetwork(new int[]{inputSize, 5, outputSize});

        countriesCSVdata = allCountriesAttributeData();

        latitudeColumn = GeneralUtils.getNumericColumn(csvCountry.LATITUDE, countriesCSVdata, 1);
        longitudeColumn = GeneralUtils.getNumericColumn(csvCountry.LONGITUDE, countriesCSVdata, 1);
        purchasingPowerColumn = GeneralUtils.getNumericColumn(csvCountry.PURCHASING_POWER, countriesCSVdata, 1);
        costOfLivingColumn = GeneralUtils.getNumericColumn(csvCountry.COST_OF_LIVING, countriesCSVdata, 1);
        affordabilityRatioColumn = purchasingPowerColumn.elementDiv(costOfLivingColumn);
        qolColumn = GeneralUtils.getNumericColumn(csvCountry.QOL, countriesCSVdata, 1);
        safetyColumn = GeneralUtils.getNumericColumn(csvCountry.SAFETY, countriesCSVdata, 1);
    }

    public double[] getInputData(String country) {
        double[] X = new double[inputSize];

        int countryID = (country.length() == 2) ? csvCountry.CODE : csvCountry.NAME;
        for (int i = 0; i < countriesCSVdata.size(); i++) {
            if (countriesCSVdata.get(i)[countryID].equals(country)) {
                //set input features with feature scaling
                X[Latitude] = latitudeColumn.get(i) / latitudeColumn.elementMaxAbs();
                X[Longitude] = longitudeColumn.get(i) / longitudeColumn.elementMaxAbs();
                X[AffordabilityRatio] = (purchasingPowerColumn.get(i) == 0.0) ? 0.0 : (affordabilityRatioColumn.get(i) / affordabilityRatioColumn.elementMaxAbs());
                X[GeneralQoL] = qolColumn.get(i) / qolColumn.elementMaxAbs();
                X[Safety] = safetyColumn.get(i) / safetyColumn.elementMaxAbs();

                break;
            }
        }

        return X;
    }
    public void run() {
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.println("Format for input is: 'latitude, longitude, purchasing power / cost of living, general quality of life, safety'\nOutput is formatted as: 'subjective quality of life, food, people and their hospitality, fun'\nEnter input (Either type the name or code of a country or use the format described above. Type STOP to terminate program): ");
            String userInput = scanner.nextLine();

            if (userInput.equalsIgnoreCase("STOP")) {
                break;
            }

            double[] X; //inputs
            String areaName;
            if (Character.isAlphabetic(userInput.charAt(0))) { //name or code of a country was typed
                areaName = userInput;
                X = getInputData(areaName);
            } else {
                areaName = "this area";
                X = new double[inputSize];
                String[] inputs = userInput.split(", ");
                for (int i = 0; i < X.length; i++) { X[i] = Double.parseDouble(inputs[i]); }
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

    public List<TrainingExample> sampleData() {
        return Arrays.asList(
                new TrainingExample(
                        getInputData("Japan"),
                        vec(9.25, 9.0, 10.0, 8.0).divide(MAX_RATING)
                ),
                new TrainingExample(
                        getInputData("United States"),
                        vec(8.0, 8.5, 7.5, 8.0).divide(MAX_RATING)
                ),
                new TrainingExample( // Bosnia stand-in
                        getInputData("Serbia"),
                        vec(7.5, 9.0, 10.0, 8.0).divide(MAX_RATING)
                ),
                new TrainingExample(
                        getInputData("Mexico"),
                        vec(5.5, 6.5, 6.5, 7.0).divide(MAX_RATING)
                ),
                new TrainingExample(
                        getInputData("Morocco"),
                        vec(6.5, 6.0, 7.0, 6.0).divide(MAX_RATING)
                ),
                new TrainingExample(
                        getInputData("Germany"),
                        vec(7.5, 7.0, 3.0, 2.5).divide(MAX_RATING)
                ),
                new TrainingExample(
                        getInputData("Afghanistan"),
                        vec(0.0, 4.0, 0.0, 0.0).divide(MAX_RATING)
                ),
                new TrainingExample(
                        getInputData("North Korea"),
                        vec(0.0, 0.0, 0.0, 0.0).divide(MAX_RATING)
                )
        );
    }

    public double[] randomInputData() {
        double[] X = new double[inputSize];
        int i = (int)(Math.random() * countriesCSVdata.size());

        //set input features with feature scaling
        X[Latitude] = latitudeColumn.get(i) / latitudeColumn.elementMaxAbs();
        X[Longitude] = longitudeColumn.get(i) / longitudeColumn.elementMaxAbs();
        X[AffordabilityRatio] = (purchasingPowerColumn.get(i) == 0.0) ? 0.0 : (affordabilityRatioColumn.get(i) / affordabilityRatioColumn.elementMaxAbs());
        X[GeneralQoL] = qolColumn.get(i) / qolColumn.elementMaxAbs();
        X[Safety] = safetyColumn.get(i) / safetyColumn.elementMaxAbs();

        return X;
    }

    public SimpleMatrix randomOutputData(double scalar) {
        return vec(Math.random(), Math.random(), Math.random(), Math.random()).scale(scalar);
    }

    public List<TrainingExample> randomizedTrainingData(int quantity, double outputScalar) {
        List<TrainingExample> trainingExamples = new ArrayList<>();
        for (int i = 0; i < quantity; i++) {
            trainingExamples.add(new TrainingExample(randomInputData(), randomOutputData(outputScalar)));
        }

        return trainingExamples;
    }

    private static List<String[]> allCountriesAttributeData() {
        try {
            FileReader fileReader = new FileReader("res/data/country_data.csv");
            CSVReader csvReader = new CSVReader(fileReader);

            return csvReader.readAll();
        } catch (Exception e) {
            e.printStackTrace();
        }

        throw new RuntimeException();
    }

    private static SimpleMatrix vec(double... data) { //column vector
        return SimpleMatrix.wrap(new DMatrixRMaj(data));
    }
}
