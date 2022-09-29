package example.tool;

public class CountryColumns {
    public final int CODE, NAME, LATITUDE, LONGITUDE, QOL, COST_OF_LIVING, PURCHASING_POWER, SAFETY;

    public CountryColumns(int CODE, int NAME, int LATITUDE, int LONGITUDE, int QOL, int COST_OF_LIVING, int PURCHASING_POWER, int SAFETY) {
        this.CODE = CODE;
        this.NAME = NAME;
        this.LATITUDE = LATITUDE;
        this.LONGITUDE = LONGITUDE;
        this.QOL = QOL;
        this.COST_OF_LIVING = COST_OF_LIVING;
        this.PURCHASING_POWER = PURCHASING_POWER;
        this.SAFETY = SAFETY;
    }
}
