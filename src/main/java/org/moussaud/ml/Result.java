package org.moussaud.ml;

public class Result {
    private String url;
    private Double muffin;
    private Double chihuahua;

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public Double getMuffin() {
        return muffin;
    }

    public void setMuffin(Double muffin) {
        this.muffin = muffin;
    }

    public Double getChihuahua() {
        return chihuahua;
    }

    public void setChihuahua(Double chihuahua) {
        this.chihuahua = chihuahua;
    }

    @Override
    public String toString() {
        return "Result [url=" + url + ", muffin=" + muffin + ", chihuahua=" + chihuahua + "]";
    }

    public String getContent() {
        return String.format("Muffin %f - Chihuahua %f", getMuffin(), getChihuahua());
    }

}
