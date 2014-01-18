package main;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import featureselectors.FeatureSelectorBaseRanker;
import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelectorRanker;
import weka.core.Instances;
import weka.core.Utils;


public class RunTests {
	List <Instances> instances;
	String[] instanceNames;
	
	FeatureSelectorBaseRanker selectionmethods[]={};
	String epoch = String.valueOf(System.currentTimeMillis()/1000);
	Boolean isWriterInitiated = false;
	PrintWriter writer;
	PrintWriter internalWriter;
	String currentInstance = "";
	
	public RunTests(List<Instances> l,FeatureSelectorBaseRanker s[], String[] instanceNames)
	{
		instances=l;
		selectionmethods=s;
		this.instanceNames = instanceNames;
	}
	
	public RunTests(List<Instances> l, String[] instanceNames){
		instances=l;
		this.instanceNames = instanceNames;
	}
	
	
	public void WriteInTotalFile(String toWrite) throws FileNotFoundException, UnsupportedEncodingException
	{
		
		if(!(new File("Result")).exists()){
			boolean success = (new File("Result")).mkdirs();
		}
		
		
		if(!isWriterInitiated){
			boolean success = (new File("Result/" + epoch)).mkdirs();
		
			
			 writer = new PrintWriter("Result/" + epoch + "/"  + "TotalResult.txt", "UTF-8");
			 isWriterInitiated = true;
		}
		
		writer.println(toWrite);
		writer.flush();
		
		if(toWrite.startsWith("++"))
		{
			if(internalWriter != null)
				internalWriter.close();
			
			if(!(new File("Result/" + epoch + "/" + currentInstance)).exists()){
				boolean success = (new File("Result/" + epoch + "/" + currentInstance)).mkdirs();
			}
			
			internalWriter = new PrintWriter("Result/" + epoch + "/" + currentInstance + "/" + toWrite.substring(2) + ".txt", "UTF-8");
		}
		else if(toWrite.startsWith("--"))
		{
			
		}
		else
		{
			internalWriter.println(toWrite);
			internalWriter.flush();
		}
		
	}
	
	public void CloseWriter()
	{
		writer.close();	
		internalWriter.close();
	}
	
	
	public void WriteInSeparateFile(String fileName, String toWrite)
	{
			
	}
	
	public void featureSelection() throws FileNotFoundException, UnsupportedEncodingException{
		
		
		
		
		Iterator<Instances> it=instances.iterator();
		
		for(int num=1;it.hasNext();num++){
			Instances inst=it.next();
			
			setCurrentInstance(num -1 );
			
			System.out.println("Performing feature selection on instance "+num+"/"+instances.size());
			
			for(int i=0;i<selectionmethods.length;i++){
				WriteInTotalFile("++"+selectionmethods[i].getName());
				selectionmethods[i].setData(inst);
				
				if(selectionmethods[i].select()){
					try {
						int selectedAttrib[]=selectionmethods[i].getSelectedAttributes();
						
						int[] selectedAttribThresh  = Arrays.copyOf(selectedAttrib, Math.min(selectedAttrib.length, selectionmethods[i].resultThreshold));
						
						WriteInTotalFile("--Selected attributes: "+Utils.arrayToString(selectedAttribThresh));
						//selectionmethods[i].printResults();
						WriteInTotalFile("--Attribute ranks: ");
						double attribrank[][]=selectionmethods[i].getAttributeRank();
						for (int j=0;j<Math.min(attribrank.length, selectionmethods[i].resultThreshold);j++){
							WriteInTotalFile(attribrank[j][0]+";"+attribrank[j][1]);
						}
					} catch (Exception e) {
						Exercise3.printErrorMsg("Could not read selected attributes");
						e.printStackTrace();
					}
				}
				else{
					Exercise3.printErrorMsg("Unknown error applying FeatureSelection Method "+selectionmethods[i].getName());
				}
				System.out.println("Done - "  + selectionmethods[i].getName() + " - " + num);
			}
			
		}
	}

	private void setCurrentInstance(int currentInstanceIndex) {
		String _currentInstanceAddr = this.instanceNames[currentInstanceIndex];

		
		String _name = _currentInstanceAddr.replace(".", "_");
		this.currentInstance = _name;
		
	}
}
