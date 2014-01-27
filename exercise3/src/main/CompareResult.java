package main;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;


import model.FeatureSelectionResult;


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
			algorithms+=inst.getName() + ", ";
			
			System.out.println("Comparing selected features of "+inst.getName()+"("+inst.getFeatures().length+")");
			
			if (num == 1){
				mutualFeatures=new ArrayList<Integer>();
				for (int i=0;i<inst.getFeatures().length;++i){
					mutualFeatures.add(inst.getFeatures()[i]);
				}
			}
			else
			{
				for (int i=0; i<mutualFeatures.size(); i++) {
					Integer e = mutualFeatures.get(i);
					if(!Arrays.asList(inst.getFeatures()).contains(e)){
						mutualFeatures.remove(e);
						i--;
					}
				}
			}
			
		}
		WriteInTotalFile("++Compared algorithms:"+algorithms);
		String listString = mutualFeatures.toString();
		listString = listString.substring(1, listString.length()-1); 
		WriteInTotalFile("++Mutual features:"+listString);
	}

}
