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

import com.sun.xml.internal.ws.util.StringUtils;

import model.FeatureSelectionResult;
import featureselectors.FeatureSelectorBaseRanker;
import featureselectors.IFeatureSelector;
import featureselectors.MyFeatureSelectorRanker;
import weka.core.Instances;
import weka.core.Utils;


public class CompareResult {
	List<FeatureSelectionResult> featureSelectionResults;
	
	List<Integer> mutualFeatures = null;
	String epoch = String.valueOf(System.currentTimeMillis()/1000);
	Boolean isWriterInitiated = false;
	PrintWriter writer;
	
	public CompareResult(List<FeatureSelectionResult> r)
	{
		featureSelectionResults=r;
	}
	
	
	public void WriteInTotalFile(String toWrite) throws FileNotFoundException, UnsupportedEncodingException
	{
		
		if(!(new File("Result")).exists()){
			boolean success = (new File("Result")).mkdirs();
		}
		
		
		if(!isWriterInitiated){
			boolean success = (new File("Result/" + epoch)).mkdirs();
		
			
			 writer = new PrintWriter("Result/" + epoch + "/"  + "Comparison.txt", "UTF-8");
			 isWriterInitiated = true;
		}
		
		writer.println(toWrite);
		writer.flush();
	}
	
	public void CloseWriter()
	{
		writer.close();	
	}
	
	public void compareFeatures() throws FileNotFoundException, UnsupportedEncodingException{
		
		Iterator<FeatureSelectionResult> it=featureSelectionResults.iterator();
		
		String algorithms = "";
		for(int num=1;it.hasNext();num++){
			FeatureSelectionResult inst=it.next();
			algorithms+=inst.Name + ", ";
			
			System.out.println("Comparing selected features of "+inst.Name+"("+inst.Features.length+")");
			
			if (num == 1)
				mutualFeatures=Arrays.asList(inst.Features);
			else
			{
				Iterator<Integer> i = mutualFeatures.iterator();
				while (i.hasNext()) {
					Integer e = i.next();
					if(!Arrays.asList(inst.Features).contains(e))
						i.remove();
				}
			}
			
		}
		WriteInTotalFile("++Compared algorithms:"+algorithms);
		String listString = mutualFeatures.toString();
		listString = listString.substring(1, listString.length()-1); 
		WriteInTotalFile("++Mutual features:"+listString);
	}

}
