package featureselectors;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

public abstract class FeatureSelectorBaseRanker implements IFeatureSelector {

	Instances data;
	AttributeSelection attsel;
	protected String name="";
	protected ASSearch searcher;
	protected ASEvaluation evaluator;
	public int resultThreshold = Integer.MAX_VALUE;
	
	public final FeatureSelectorBaseRanker setFeatureSelectorName(String name){
		this.name = name;
		return this;
	}
	
	public final FeatureSelectorBaseRanker setSearcher(ASSearch searcher)
	{
		this.searcher = searcher;
		return this;
	}
	
	public final FeatureSelectorBaseRanker setEvaluator(ASEvaluation evaluator){
		this.evaluator = evaluator;
		return this;
	}
	
	public final FeatureSelectorBaseRanker setResultThreshold(int threshold){
		this.resultThreshold = threshold;
		return this;
	}

	@Override
	public final void setData(Instances data) {
		this.data = data;
	}

	@Override
	public boolean select() {
		return true;
	}

	@Override
	public final int[] getSelectedAttributes() throws Exception {
		return attsel.selectedAttributes() ;
	}

	@Override
	public final String getName() {
		return name;
	}

	@Override
	public final double[][] getAttributeRank() throws Exception {
		//attsel.setRanking(true);
		return attsel.rankedAttributes();
	}
	@Override
	public final void printResults(){
		attsel.toResultsString();
	}

}
