package main;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.LinkedList;
import java.util.List;

import model.FeatureSelectionResult;
import featureselectors.FeatureSelectorBaseRanker;
import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelectorRanker;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.HoldOutSubsetEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.LatentSemanticAnalysis;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;
import weka.core.Utils;
import input.CMDReader;
import input.InstanceReader;
import input.ResultReader;

public class Exercise3 {

	/**
	 * @param args
	 * @throws UnsupportedEncodingException
	 * @throws FileNotFoundException
	 */
	public static void main(String[] args) throws FileNotFoundException,
			UnsupportedEncodingException {
		// TODO Auto-generated method stub
		System.out.println("Started");

		CMDReader cmd = new CMDReader();
		cmd.parse(args);
		if (cmd.isListTechniques()) {
			printTechniques();
		} else if (!cmd.isDirSet()) {
			printErrorMsg("Instance directory not set");
			cmd.printUsage();
		} else if (cmd.useFeature()) {
			runWithFeatures(cmd.getFeatures(), cmd.getInstanceDir());
		}
		else if (cmd.compareResult()) {
			runComparisonOfResultFiles(cmd.getInstanceDir());
		}
		else
			runWithFeatures(null, cmd.getInstanceDir());

	}

	public static void printErrorMsg(String error) {
		System.err.println("Err: " + error);

	}

	public static void printTechniques() {
		System.out.format("%s%10s%10s", "name", "Search", "Eval");
	}

	public static void runWithFeatures(String[] features, String instancedir) {
		if (features == null) {
			InstanceReader ir = new InstanceReader(instancedir);

			List<Instances> l = null;
			try {
				l = ir.read();
			} catch (Exception e) {
				printErrorMsg("Could not read instances");
				printErrorMsg(e.getLocalizedMessage());
			}

			List<FeatureSelectorBaseRanker> _selectorBases = new LinkedList<FeatureSelectorBaseRanker>();
			
				
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("PrincipalComponents-Ranker")
				.setEvaluator(new PrincipalComponents())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("LatentSemanticAnalysis-Ranker")
				.setEvaluator(new LatentSemanticAnalysis())
				.setSearcher(new Ranker()).setResultThreshold(50));
			
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("SymmetricalUncertAttribute-Ranker")
				.setEvaluator(new SymmetricalUncertAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("OneRAttribute-Ranker")
				.setEvaluator(new OneRAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("GainRatioAttribute-Ranker")
				.setEvaluator(new GainRatioAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));
	
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("CfsSubset-BestFirst")
				.setEvaluator(new CfsSubsetEval())
				.setSearcher(new BestFirst()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("InfoGain-Ranker")
				.setEvaluator(new InfoGainAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName("ReliefFAttribute-Ranker")
					.setEvaluator(new ReliefFAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			
			
			String[] _instanceNames = ir.instaceAddresses
					.toArray(new String[] {});
			RunTests rt = new RunTests(l,
					_selectorBases.toArray(new FeatureSelectorBaseRanker[] {}),
					_instanceNames);
			try {
				rt.featureSelection();
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}
	
	public static void runComparisonOfResultFiles(String instancedir) {
		ResultReader resultReader=new ResultReader(instancedir);
		List<FeatureSelectionResult> r=null;
		try {
			r = resultReader.read();
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		CompareResult cr = new CompareResult(r);
		try {
			cr.compareFeatures();
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

}
