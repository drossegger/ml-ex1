package main;

import java.util.List;

import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelector;

import weka.core.Instances;
import weka.core.Utils;
import input.CMDReader;
import input.InstanceReader;

public class Exercise3 {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		CMDReader cmd=new CMDReader();
		cmd.parse(args);
		if(!cmd.isDirSet()){
			printErrorMsg("Instance directory not set");
			cmd.printUsage();
		}
		else{
			InstanceReader ir=new InstanceReader(cmd.getInstanceDir());
			List<Instances> l = null;
			try {
				l=ir.read();
			} catch (Exception e) {
				printErrorMsg("Could not read instances");
				printErrorMsg(e.getLocalizedMessage());
			}
			//IFeatureSelector fs[]=new IFeatureSelector[l.size()];
			for(Instances i : l){
				IFeatureSelector fs=new MyFeatureSelector();
				fs.setData(i);
				if(fs.select()){
					try {
						int selectedAttrib[]=fs.getSelectedAttributes();
						System.out.println(Utils.arrayToString(selectedAttrib));
					} catch (Exception e) {
						printErrorMsg("Could not read selected attributes");
						
					}
					
				}
			}
		}
		
			
	}
	public static void printErrorMsg(String error){
		System.err.println("Err: "+error);
		
	}

}
