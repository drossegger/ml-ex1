package featureselectors;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.core.Instance;
import weka.core.Instances;

public class MyFeatureSelector implements IFeatureSelector {
	Instances data;
	AttributeSelection attsel;

	@Override
	public void setData(Instances data) {
		this.data = data;
	}

	@Override
	public boolean select() {
		attsel = new AttributeSelection();
															
		//CfsSubsetEval eval = new CfsSubsetEval();
		//GreedyStepwise search = new GreedyStepwise();
		//search.setSearchBackwards(true);
		attsel.setEvaluator(new GainRatioAttributeEval());
		attsel.setSearch(new Ranker());
		attsel.setRanking(true);
		try {
			attsel.SelectAttributes(data);
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	@Override
	public int[] getSelectedAttributes() throws Exception {
		return attsel.selectedAttributes() ;
	}

	@Override
	public String getName() {
		return "MyFeatureSelectionAlgorithm";
	}

	@Override
	public double[][] getAttributeRank() throws Exception {
		//attsel.setRanking(true);
		return attsel.rankedAttributes();
	}
	@Override
	public void printResults(){
		attsel.toResultsString();
	}

}
