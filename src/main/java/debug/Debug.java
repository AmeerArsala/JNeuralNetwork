package debug;

import java.io.PrintStream;
import java.util.List;

public class Debug {

    public static <T> void printAll(List<T> list, PrintStream printStream) {
        for (T item : list) {
            printStream.println(item);
        }
    }

    public static <T> void printAll(T[] list, PrintStream printStream) {
        for (T item : list) {
            printStream.println(item);
        }
    }

}
