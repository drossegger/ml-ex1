package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.UnsupportedEncodingException;
import java.util.LinkedList;
import java.util.List;

import model.FeatureSelectionResult;
import featureselectors.FeatureSelectorBaseRanker;
import featureselectors.MyFeatureSelectorRanker;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.ConsistencySubsetEval;
import weka.attributeSelection.FilteredAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.LatentSemanticAnalysis;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;
import input.CMDReader;
import input.InstanceReader;
import input.ResultReader;

public class Exercise3 {
	static int topn=10;
	static float thresh=0.5f;
	/**
	 * @param args
	 * @throws UnsupportedEncodingException
	 * @throws FileNotFoundException
	 */
	public static void main(String[] args) throws FileNotFoundException,
			UnsupportedEncodingException {
		CMDReader cmd = new CMDReader();
		cmd.parse(args);
		try {
			if (cmd.isHelp()) {
				cmd.printUsage();
				return;
			}
			if(cmd.isTopN())topn=cmd.getTopN();
			if(cmd.isAttThresh())thresh=cmd.getAttThresh();
			if (cmd.isListTechniques()) {
				printTechniques();
			} else if (!cmd.isDirSet()) {
				printErrorMsg("Instance directory not set");
				cmd.printUsage();
				return;
			} else if (cmd.useFeature()) {
				runWithFeatures(cmd.getFeatures(), cmd.getInstanceDir());
			} else if (cmd.compareResult()) {
				runComparisonOfResultFiles(cmd.getInstanceDir());
			} else
				runWithFeatures(null, cmd.getInstanceDir());
		} catch (NullPointerException e) {
		}
	}

	public static void printErrorMsg(String error) {
		System.err.println("Err: " + error);

	}

	public static void printTechniques() {
		System.out.format("%1$15s%2$30s%3$50s\n", "name", "Search", "Eval");
		System.out.format("%1$15s%2$30s%3$50s\n", "sua-ranker", "Ranker",
				"SelectedUncertAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "oner-ranker", "Ranker",
				"OneRAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "gainr-ranker", "Ranker",
				"GainRAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "ig-ranker", "Ranker",
				"InfoGainAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "rel-ranker", "Ranker",
				"ReliefAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "rel-ranker", "Ranker",
				"ReliefAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "cfs-greedy", "GreedyStepWise",
				"CfsSubset");
		System.out.format("%1$15s%2$30s%3$50s\n", "chis-ranker", "Ranker",
				"ChiSquaredAttribute");
		System.out.format("%1$15s%2$30s%3$50s\n", "cons-bestfirst", "BestFirst",
				"ConsistencySubset");
		System.out.format("%1$15s%2$30s%3$50s\n", "pca-ranker", "Ranker",
				"PrincipalComponents");
		System.out.format("%1$15s%2$30s%3$50s\n", "lsa-ranker", "Ranker",
				"LatentSemanticAnalysis");
		

	}

	public static void runWithFeatures(String[] features, String instancedir) {
		InstanceReader ir = new InstanceReader(instancedir);
		List<FeatureSelectorBaseRanker> _selectorBases = new LinkedList<FeatureSelectorBaseRanker>();
		List<Instances> l = null;
		try {
			l = ir.read();
		} catch (Exception e) {
			printErrorMsg("Could not read instances");
			printErrorMsg(e.getLocalizedMessage());
		}

		if (features == null) {
			_selectorBases
					.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"CfsSubset-GreedyStepwise-Supervised")
							.setEvaluator(
									new CfsSubsetEval())
							.setSearcher(new GreedyStepwise())
							.setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"ChiSquaredAttribute-Ranker-Supervised")
					.setEvaluator(new ChiSquaredAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"ConsistencySubset-BestFirst-Supervised")
					.setEvaluator(new ConsistencySubsetEval())
					.setSearcher(new BestFirst()).setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"GainRatioAttribute-Ranker-Supervised")
					.setEvaluator(new GainRatioAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"InfoGain-Ranker-Supervised")
					.setEvaluator(new InfoGainAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"OneRAttribute-Ranker-Supervised")
					.setEvaluator(new OneRAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"ReliefFAttribute-Ranker-Supervised")
					.setEvaluator(new ReliefFAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			_selectorBases
					.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"SymmetricalUncertAttribute-Ranker-Supervised")
							.setEvaluator(
									new SymmetricalUncertAttributeEval())
							.setSearcher(new Ranker())
							.setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"PrincipalComponents-Ranker-Unsupervised")
					.setEvaluator(new PrincipalComponents())
					.setSearcher(new Ranker()).setResultThreshold(50));
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"LatentSemanticAnalysis-Ranker-Unsupervised")
					.setEvaluator(new LatentSemanticAnalysis())
					.setSearcher(new Ranker()).setResultThreshold(50));

			/*
			// unsupervised
			FilteredAttributeEval f_SymmetricalUncertAttribute = new FilteredAttributeEval();
			f_SymmetricalUncertAttribute
					.setAttributeEvaluator(new SymmetricalUncertAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"SymmetricalUncertAttribute-Ranker-Unsupervised")
					.setEvaluator(f_SymmetricalUncertAttribute)
					.setSearcher(new Ranker()).setResultThreshold(50));

			FilteredAttributeEval f_OneRAttributeEval = new FilteredAttributeEval();
			f_OneRAttributeEval.setAttributeEvaluator(new OneRAttributeEval());
			_selectorBases
					.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"OneRAttribute-Ranker-Unsupervised")
							.setEvaluator(f_OneRAttributeEval)
							.setSearcher(new Ranker()).setResultThreshold(50));

			FilteredAttributeEval f_GainRatioAttributeEval = new FilteredAttributeEval();
			f_GainRatioAttributeEval
					.setAttributeEvaluator(new GainRatioAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"GainRatioAttribute-Ranker-Unsupervised")
					.setEvaluator(f_GainRatioAttributeEval)
					.setSearcher(new Ranker()).setResultThreshold(50));

			FilteredAttributeEval f_InfoGainAttributeEval = new FilteredAttributeEval();
			f_InfoGainAttributeEval
					.setAttributeEvaluator(new InfoGainAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName("InfoGain-Ranker-Unsupervised")
					.setEvaluator(f_InfoGainAttributeEval)
					.setSearcher(new Ranker()).setResultThreshold(50));

			FilteredAttributeEval f_ReliefFAttributeEval = new FilteredAttributeEval();
			f_ReliefFAttributeEval
					.setAttributeEvaluator(new ReliefFAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName(
							"ReliefFAttribute-Ranker-Unspervised")
					.setEvaluator(f_ReliefFAttributeEval)
					.setSearcher(new Ranker()).setResultThreshold(50));

			*/
		} else {
			for (String s : features) {
				if (s.equals("cfs-greedy"))
					_selectorBases
							.add(new MyFeatureSelectorRanker()
									.setFeatureSelectorName(
											"CfsSubset-GreedyStepwise-Supervised")
									.setEvaluator(
											new CfsSubsetEval())
									.setSearcher(new GreedyStepwise())
									.setResultThreshold(50));
				else if (s.equals("chis-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"ChiSquaredAttribute-Ranker-Supervised")
							.setEvaluator(new ChiSquaredAttributeEval())
							.setSearcher(new Ranker()).setResultThreshold(50));
				else if (s.equals("cons-bestfirst"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"ConsistencySubset-BestFirst-Supervised")
							.setEvaluator(new ConsistencySubsetEval())
							.setSearcher(new BestFirst()).setResultThreshold(50));
				else if (s.equals("gainr-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"GainRatioAttribute-Ranker-Supervised")
							.setEvaluator(new GainRatioAttributeEval())
							.setSearcher(new Ranker()).setResultThreshold(50));
				else if (s.equals("ig-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"InfoGain-Ranker-Supervised")
							.setEvaluator(new InfoGainAttributeEval())
							.setSearcher(new Ranker()).setResultThreshold(50));
				else if (s.equals("oner-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"OneRAttribute-Ranker-Supervised")
							.setEvaluator(new OneRAttributeEval())
							.setSearcher(new Ranker()).setResultThreshold(50));
				else if (s.equals("rel-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"ReliefFAttribute-Ranker-Supervised")
							.setEvaluator(new ReliefFAttributeEval())
							.setSearcher(new Ranker()).setResultThreshold(50));
				else if (s.equals("sua-ranker"))
					_selectorBases
							.add(new MyFeatureSelectorRanker()
									.setFeatureSelectorName(
											"SymmetricalUncertAttribute-Ranker-Supervised")
									.setEvaluator(
											new SymmetricalUncertAttributeEval())
									.setSearcher(new Ranker())
									.setResultThreshold(50));
				else if (s.equals("pca-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"PrincipalComponents-Ranker-Unsupervised")
							.setEvaluator(new PrincipalComponents())
							.setSearcher(new Ranker()).setResultThreshold(50));
				else if (s.equals("lsa-ranker"))
					_selectorBases.add(new MyFeatureSelectorRanker()
							.setFeatureSelectorName(
									"LatentSemanticAnalysis-Ranker-Unsupervised")
							.setEvaluator(new LatentSemanticAnalysis())
							.setSearcher(new Ranker()).setResultThreshold(50));
			}

		}
		String[] _instanceNames = ir.getInstanceAddresses()
				.toArray(new String[] {});
		RunTests rt = new RunTests(l,
				_selectorBases.toArray(new FeatureSelectorBaseRanker[] {}),
				_instanceNames);
		try {
			rt.featureSelection();
			CompareResult cr = new CompareResult();
			for (int i = 0; i < rt.getResults().length; ++i) {
				cr.compareFeatures(rt.getResults()[i], "");
			}

		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}

	}

	public static void runComparisonOfResultFiles(String instancedir) {
		File path = new File(instancedir);
		ResultReader resultReader = new ResultReader();
		CompareResult cr = new CompareResult();

		for (final File file : path.listFiles(new FilenameFilter() {
			@Override
			public boolean accept(File dir, String name) {
				return name.toLowerCase().endsWith(".txt");
			}
		})) {
			if (file.isFile()) {
				List<FeatureSelectionResult> r = null;
				try {
					r = resultReader.read(file.getAbsolutePath());
				} catch (Exception e1) {
					e1.printStackTrace();
				}

				String fileName = file.getName().replaceFirst("[.][^.]+$", "");
				try {
					cr.compareFeatures(r, fileName);
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					e.printStackTrace();
				}
			}
		}
	}
}
