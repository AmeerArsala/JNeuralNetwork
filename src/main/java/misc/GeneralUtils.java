package misc;

import neuralnetwork.util.Operations;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

public class GeneralUtils {
    //assumes the first row consists of labels
    public static String[] getColumn(int col, List<String[]> data, int startRow) {
        int length = data.size();
        String[] column = new String[length];

        for (int i = startRow; i < length; ++i) {
            column[i - startRow] = data.get(i)[col];
        }

        return column;
    }

    //assumes the first row consists of labels
    public static SimpleMatrix getNumericColumn(int col, List<String[]> data, int startRow) {
        int length = data.size();
        double[] column = new double[length];

        for (int i = startRow; i < length; ++i) {
            column[i - startRow] = Double.parseDouble(data.get(i)[col]);
        }

        return Operations.colVector(column);
    }
}
