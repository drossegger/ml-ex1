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
import weka.attributeSelection.FilteredAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;
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
			
			//supervised
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("SymmetricalUncertAttribute-Ranker-Supervised")
				.setEvaluator(new SymmetricalUncertAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("OneRAttribute-Ranker-Supervised")
				.setEvaluator(new OneRAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("GainRatioAttribute-Ranker-Supervised")
				.setEvaluator(new GainRatioAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("InfoGain-Ranker-Supervised")
				.setEvaluator(new InfoGainAttributeEval())
				.setSearcher(new Ranker()).setResultThreshold(50));

			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName("ReliefFAttribute-Ranker-Supervised")
					.setEvaluator(new ReliefFAttributeEval())
					.setSearcher(new Ranker()).setResultThreshold(50));
			
			//unsupervised
			FilteredAttributeEval f_SymmetricalUncertAttribute = new FilteredAttributeEval();
			f_SymmetricalUncertAttribute.setAttributeEvaluator(new SymmetricalUncertAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("SymmetricalUncertAttribute-Ranker-Unsupervised")
				.setEvaluator(f_SymmetricalUncertAttribute)
				.setSearcher(new Ranker()).setResultThreshold(50));
			
			FilteredAttributeEval f_OneRAttributeEval = new FilteredAttributeEval();
			f_OneRAttributeEval.setAttributeEvaluator(new OneRAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("OneRAttribute-Ranker-Unsupervised")
				.setEvaluator(f_OneRAttributeEval)
				.setSearcher(new Ranker()).setResultThreshold(50));
	
			FilteredAttributeEval f_GainRatioAttributeEval = new FilteredAttributeEval();
			f_GainRatioAttributeEval.setAttributeEvaluator(new GainRatioAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("GainRatioAttribute-Ranker-Unsupervised")
				.setEvaluator(f_GainRatioAttributeEval)
				.setSearcher(new Ranker()).setResultThreshold(50));
	
			FilteredAttributeEval f_InfoGainAttributeEval = new FilteredAttributeEval();
			f_InfoGainAttributeEval.setAttributeEvaluator(new InfoGainAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
				.setFeatureSelectorName("InfoGain-Ranker-Unsupervised")
				.setEvaluator(f_InfoGainAttributeEval)
				.setSearcher(new Ranker()).setResultThreshold(50));
	
			FilteredAttributeEval f_ReliefFAttributeEval = new FilteredAttributeEval();
			f_ReliefFAttributeEval.setAttributeEvaluator(new ReliefFAttributeEval());
			_selectorBases.add(new MyFeatureSelectorRanker()
					.setFeatureSelectorName("ReliefFAttribute-Ranker-Unspervised")
					.setEvaluator(f_ReliefFAttributeEval)
					.setSearcher(new Ranker()).setResultThreshold(50));
			
			String[] _instanceNames = ir.instaceAddresses
					.toArray(new String[] {});
			RunTests rt = new RunTests(l,
					_selectorBases.toArray(new FeatureSelectorBaseRanker[] {}),
					_instanceNames);
			try {
				rt.featureSelection();
				CompareResult cr=new CompareResult();
				for(int i=0; i<rt.getResults().length;++i){
					cr.compareFeatures(rt.getResults()[i], "");
				}
				
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}
	
	public static void runComparisonOfResultFiles(String instancedir) {
		File path = new File(instancedir);
		ResultReader resultReader=new ResultReader();
		CompareResult cr = new CompareResult();
		
		for (final File file : path.listFiles(new FilenameFilter() {
			@Override
			public boolean accept(File dir, String name) {
				return name.toLowerCase().endsWith(".txt");
			}
		})) {
			if (file.isFile()) {
				List<FeatureSelectionResult> r=null;
				try {
					r = resultReader.read(file.getAbsolutePath());
				} catch (Exception e1) {
					e1.printStackTrace();
				}

				String fileName=file.getName().replaceFirst("[.][^.]+$", "");
				try {
					cr.compareFeatures(r, fileName);
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					e.printStackTrace();
				}
		    }
		}
	}
}
