package main;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.LinkedList;
import java.util.List;

import featureselectors.FeatureSelectorBaseRanker;
import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelectorRanker;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.HoldOutSubsetEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.core.Instances;
import weka.core.Utils;
import input.CMDReader;
import input.InstanceReader;

public class Exercise3 {

	/**
	 * @param args
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		// TODO Auto-generated method stub
		System.out.println("Started");
		
		CMDReader cmd=new CMDReader();
		cmd.parse(args);
		if(!cmd.isDirSet()){
	    		printErrorMsg("Instance directory not set");
			cmd.printUsage();
		}
		else
		{
			InstanceReader ir=new InstanceReader(cmd.getInstanceDir());
			
			
			List<Instances> l = null;
			try {
				l=ir.read();
			} catch (Exception e) {
				printErrorMsg("Could not read instances");
				printErrorMsg(e.getLocalizedMessage());
			}
			
			List<FeatureSelectorBaseRanker> _selectorBases = new LinkedList<FeatureSelectorBaseRanker>();
			
			_selectorBases.add(new MyFeatureSelectorRanker()
									.setFeatureSelectorName("ReliefFAttributeEval-Ranker")
									.setEvaluator(new ReliefFAttributeEval())
									.setSearcher(new Ranker())
									.setResultThreshold(50));
			
			
			_selectorBases.add(new MyFeatureSelectorRanker()
									.setFeatureSelectorName("InfoGain-Ranker")
									.setEvaluator(new InfoGainAttributeEval())
									.setSearcher(new Ranker())
									.setResultThreshold(50));
			
			
			String[] _instanceNames = ir.instaceAddresses.toArray(new String[]{});
			RunTests rt = new RunTests(l,_selectorBases.toArray(new FeatureSelectorBaseRanker[]{}) ,_instanceNames);
			rt.featureSelection();
		}
		
			
	}
	public static void printErrorMsg(String error){
		System.err.println("Err: "+error);
		
	}

}
