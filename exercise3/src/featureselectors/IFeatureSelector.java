package featureselectors;

import weka.core.Instances;

public interface IFeatureSelector {
	public void setData(Instances data);
	public boolean select();
	public int[] getSelectedAttributes() throws Exception;
	public double[][] getAttributeRank() throws Exception;
	public String getName();
	void printResults();
}
