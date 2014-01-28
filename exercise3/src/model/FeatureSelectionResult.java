package model;

public class FeatureSelectionResult {
	private String name;
	private int[] features;
	private float[] weights;
	
	public FeatureSelectionResult(){
	
	}
	public FeatureSelectionResult(String n, int[] f){
		setName(n);
		setFeatures(f);
		
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int[] getFeatures() {
		return features;
	}
	public void setFeatures(int[] features) {
		this.features = features;
	}
	public float[] getWeights() {
		return weights;
	}
	public void setWeights(float[] weights) {
		this.weights = weights;
	}
}
