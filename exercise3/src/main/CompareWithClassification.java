package main;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import javax.swing.event.ListSelectionEvent;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import model.FeatureSelectionResult;

public class CompareWithClassification {
	
	public List<List<Double>> compareFeatures(List<FeatureSelectionResult> r, Instances data, int topN, float threshold) throws Exception{
		{
			LinkedList<List<Double>> _output = new LinkedList<List<Double>>();
			
			 // setting class attribute if the data format does not provide this information
			 // For example, the XRFF format saves the class attribute information as well
			 if (data.classIndex() == -1)
			   data.setClassIndex(data.numAttributes() - 1);
			 
				//Random
				{
					int _allAttr = data.numAttributes();
					

					Classifier cls = new weka.classifiers.lazy.IBk();
					
					List<Integer> _includeAttributes = new LinkedList<Integer>();
					for(int i =0;i< 50 ; i++){
						int _selected = (int)(Math.random() * _allAttr);
						_includeAttributes.add(_selected);
					}
					
					Collections.sort(_includeAttributes);
					
					
					Instances _localTrain = data.trainCV(4, 1); // use 75% of data for training
				    Instances _localTest = data.testCV(4, 1);
				    
				    for(int j = _localTrain.numAttributes()-2 ; j >= 0;j--){
				    	boolean _delete = true;
				    	for(int k = 0 ; k < _includeAttributes.size() ; k++){
				    		
				    		if(_includeAttributes.get(k) == j){
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
					//System.out.println(eval.weightedFMeasure());
			
					LinkedList<Double> _localResults = new LinkedList<Double>();
					_localResults.add(eval.weightedPrecision());
					_localResults.add(eval.weightedRecall());
					_localResults.add(eval.weightedFMeasure());
					 
					_output.add(_localResults);
				
				}
			 
			// 10 classifications for each attrib selection result
			for(int i = 0 ; i < r.size();i++){
				
				Classifier cls = new weka.classifiers.lazy.IBk();
				
				FeatureSelectionResult _f = r.get(i);
				int[] _res = _f.getFeatures();
				
				Arrays.sort(_res);
				
				Instances _localTrain = data.trainCV(4, 1); // use 75% of data for training
			    Instances _localTest = data.testCV(4, 1);
			    
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
				//System.out.println(eval.weightedFMeasure());
		
				LinkedList<Double> _localResults = new LinkedList<Double>();
				_localResults.add(eval.weightedPrecision());
				_localResults.add(eval.weightedRecall());
				_localResults.add(eval.weightedFMeasure());
				 
				_output.add(_localResults);
				
				
			}
			
			//intersection
			{
				int _allAttr = data.numAttributes();
				
	
				int [] _attrCount = new int[_allAttr];
						
				for(int i = 0 ; i < r.size();i++){
					
					
					FeatureSelectionResult _f = r.get(i);
					int[] _res = _f.getFeatures();
					
					for(int j = 0 ; j< _res.length;j++){
						_attrCount[_res[j]]++;
					}
					
					
				}
				
				Classifier cls = new weka.classifiers.lazy.IBk();
				
				List<Integer> _includeAttributes = new LinkedList<Integer>();
				
				for(int i =0;i< _allAttr ; i++){
					float _temp = (float)_attrCount[i]/r.size();
					if(_temp>= threshold)
						_includeAttributes.add(i);
				}
				
				Collections.sort(_includeAttributes);
				
				
				Instances _localTrain = data.trainCV(4, 1); // use 75% of data for training
			    Instances _localTest = data.testCV(4, 1);
			    
			    for(int j = _localTrain.numAttributes()-2 ; j >= 0;j--){
			    	boolean _delete = true;
			    	for(int k = 0 ; k < _includeAttributes.size() ; k++){
			    		
			    		if(_includeAttributes.get(k) == j){
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
				//System.out.println(eval.weightedFMeasure());
		
				LinkedList<Double> _localResults = new LinkedList<Double>();
				_localResults.add(eval.weightedPrecision());
				_localResults.add(eval.weightedRecall());
				_localResults.add(eval.weightedFMeasure());
				 
				_output.add(_localResults);
			
			}
			
			//union
			{

				Classifier cls = new weka.classifiers.lazy.IBk();
				
				List<Integer> _includeAttributes = new LinkedList<Integer>();
				
				for(int i = 0 ; i < r.size();i++){
					
					FeatureSelectionResult _f = r.get(i);
					int[] _res = _f.getFeatures();
					
					for(int j=0;j< _res.length && j< topN; j++){
						if(!_includeAttributes.contains(_res[j])){
							_includeAttributes.add(_res[j]);
						}
					}
										
				}
				
				Collections.sort(_includeAttributes);
				Instances _localTrain = data.trainCV(4, 1); // use 75% of data for training
			    Instances _localTest = data.testCV(4, 1);
			    
			    for(int j = _localTrain.numAttributes()-2 ; j >= 0;j--){
			    	boolean _delete = true;
			    	for(int k = 0 ; k < _includeAttributes.size() ; k++){
			    		
			    		if(_includeAttributes.get(k) == j){
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
				//System.out.println(eval.weightedFMeasure());
		
				LinkedList<Double> _localResults = new LinkedList<Double>();
				_localResults.add(eval.weightedPrecision());
				_localResults.add(eval.weightedRecall());
				_localResults.add(eval.weightedFMeasure());
				 
				_output.add(_localResults);
				
				
			}
			
			for(int i= 0 ; i< _output.size();i++){
				System.out.println(_output.get(i).get(0).toString() + ", " + _output.get(i).get(1).toString() + ", " + _output.get(i).get(2).toString());
			}
			
			return _output;
		}
	}

	
}
