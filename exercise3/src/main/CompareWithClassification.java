package main;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import model.FeatureSelectionResult;

public class CompareWithClassification {
	
	public List<List<Double>> compareFeatures(List<FeatureSelectionResult> r, Instances data, int topN, float threshold) throws Exception{
		{
			LinkedList<List<Double>> _output = new LinkedList<List<Double>>();
			
			 // setting class attribute if the data format does not provide this information
			 // For example, the XRFF format saves the class attribute information as well
			 if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes() - 1);
			 
			 
			 Instances train = data.trainCV(4, 0); // use 75% of data for training
			 Instances test = data.testCV(4, 0); //use 25% of the data for testing
			 
			 // train classifier
			 
			 
			 {
				 Classifier cls = new weka.classifiers.lazy.IBk();
				 
				 Instances _allTrain = data.trainCV(4, 0); // use 75% of data for training
				 Instances _allTest = data.testCV(4, 0);
				 
				 
				 //without anything
				 cls.buildClassifier(_allTrain);
				 // evaluate classifier and print some statistics
				 Evaluation eval = new Evaluation(_allTrain);
				 eval.evaluateModel(cls, _allTest);
				 System.out.println(eval.weightedFMeasure());
				 
				 LinkedList<Double> _results = new LinkedList<Double>();
				 
				 _results.add(eval.weightedPrecision());
				 _results.add(eval.weightedRecall());
				 _results.add(eval.weightedFMeasure());
				 
				 _output.add(_results);
			 }
			 
			
			for(int i = 0 ; i < r.size();i++){
				
				Classifier cls = new weka.classifiers.lazy.IBk();
				
				FeatureSelectionResult _f = r.get(i);

				
				
				
				int[] _res = _f.getFeatures();
				
				
				 
				 
				Instances _localTrain = data.trainCV(4, 0); // use 75% of data for training
			    Instances _localTest = data.testCV(4, 0);
			    
			    for(int j = _localTrain.numAttributes()-2 ; j >= 0;j--){
			    	boolean _delete = true;
			    	for(int k = 0 ; k < _res.length ; k++){
			    		
			    		if(_res[k] == j){
			    			_delete = false;
			    		}

			    	}
		    		if(_delete){
		    			_localTrain.deleteAttributeAt(j);
		    			_localTest.deleteAttributeAt(j);
		    		}
			    }
					 

				
				 //with attribute selection
				cls.buildClassifier(_localTrain);
				// evaluate classifier and print some statistics
				Evaluation eval = new Evaluation(_localTrain);
				eval.evaluateModel(cls, _localTest);
				System.out.println(eval.weightedFMeasure());
		
				LinkedList<Double> _localResults = new LinkedList<Double>();
				_localResults.add(eval.weightedPrecision());
				_localResults.add(eval.weightedRecall());
				_localResults.add(eval.weightedFMeasure());
				 
				_output.add(_localResults);
				
				
			}
			
			
			return _output;
		}
	}

	
}
