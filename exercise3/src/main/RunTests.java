package main;
import java.util.Iterator;
import java.util.List;

import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelector;

import weka.core.Instances;
import weka.core.Utils;


public class RunTests {
	List <Instances> instances;
	IFeatureSelector selectionmethods[]={new MyFeatureSelector()};
	
	public RunTests(List<Instances> l,IFeatureSelector s[]){
		instances=l;
		selectionmethods=s;
	}
	
	public RunTests(List<Instances> l){
		instances=l;
	}
	
	public void featureSelection(){
		Iterator<Instances> it=instances.iterator();
		
		for(int num=1;it.hasNext();num++){
			Instances inst=it.next();
			
			System.out.println("Performing feature selection on instance "+num+"/"+instances.size());
			for(int i=0;i<selectionmethods.length;i++){
				System.out.println("	Performing selection method "+selectionmethods[i].getName());
				selectionmethods[i].setData(inst);
				
				if(selectionmethods[i].select()){
					try {
						int selectedAttrib[]=selectionmethods[i].getSelectedAttributes();
						System.out.println("	Selected attributes: "+Utils.arrayToString(selectedAttrib));
					} catch (Exception e) {
						Exercise3.printErrorMsg("Could not read selected attributes");
					}
				}
				else{
					Exercise3.printErrorMsg("Unknown error applying FeatureSelection Method "+selectionmethods[i].getName());
				}
				
			}
			
		}
	}
}
