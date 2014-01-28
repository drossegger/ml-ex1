package main;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import model.FeatureSelectionResult;

import featureselectors.FeatureSelectorBaseRanker;
import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelectorRanker;
import weka.core.Instances;
import weka.core.Utils;


public class RunTests {
	List <Instances> instances;
	String[] instanceNames;
	private List <FeatureSelectionResult>[] results;
	public List<FeatureSelectionResult>[] getResults() {
		return results;
	}

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
		results=(List<FeatureSelectionResult>[])new List[l.size()];
	}
	
	public RunTests(List<Instances> l, String[] instanceNames){
		instances=l;
		this.instanceNames = instanceNames;
		results=(List<FeatureSelectionResult>[])new List[l.size()];
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
			results[num-1]=new ArrayList<FeatureSelectionResult>();
			setCurrentInstance(num -1 );
			
			System.out.println("Performing feature selection on instance "+num+"/"+instances.size());
			
			for(int i=0;i<selectionmethods.length;i++){
				WriteInTotalFile("++"+selectionmethods[i].getName());
				selectionmethods[i].setData(inst);
				
				if(selectionmethods[i].select()){
					try {
						int selectedAttrib[]=selectionmethods[i].getSelectedAttributes();
						
						int[] selectedAttribThresh  = Arrays.copyOf(selectedAttrib, Math.min(selectedAttrib.length, selectionmethods[i].resultThreshold));
						results[num-1].add(new FeatureSelectionResult(selectionmethods[i].getName(),selectedAttribThresh));
						WriteInTotalFile("--Selected attributes: "+Utils.arrayToString(selectedAttribThresh));
						//selectionmethods[i].printResults();
						WriteInTotalFile("--Attribute ranks: ");
						double attribrank[][]=null;
						try{
							attribrank = selectionmethods[i].getAttributeRank();
						} catch (Exception e) {
							Exercise3.printErrorMsg("No ranking available");
						}
						if (attribrank != null)
						{
							for (int j=0;j<Math.min(attribrank.length, selectionmethods[i].resultThreshold);j++){
								WriteInTotalFile(attribrank[j][0]+";"+attribrank[j][1]);
							}
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
			int k = 0;
			
		}
	}

	private void setCurrentInstance(int currentInstanceIndex) {
		String _currentInstanceAddr = this.instanceNames[currentInstanceIndex];

		
		String _name = _currentInstanceAddr.replace(".", "_");
		this.currentInstance = _name;
		
	}
}
