package featureselectors;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.core.Instance;
import weka.core.Instances;

public class MyFeatureSelectorRanker extends FeatureSelectorBaseRanker {
	
	@Override
	public boolean select() {
		attsel = new AttributeSelection();
														
		attsel.setEvaluator(this.evaluator);
		attsel.setSearch(this.searcher);
		attsel.setRanking(true);
		try {
			attsel.SelectAttributes(data);
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}
}
