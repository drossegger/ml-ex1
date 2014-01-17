package featureselectors;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
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
															
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		search.setSearchBackwards(true);
		attsel.setEvaluator(eval);
		attsel.setSearch(search);
		try {
			attsel.SelectAttributes(data);
		} catch (Exception e) {
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
		// TODO Auto-generated method stub
		return "MyFeatureSelectionAlgorithm";
	}

}
