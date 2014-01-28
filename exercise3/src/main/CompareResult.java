package main;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;




import model.FeatureSelectionResult;


public class CompareResult {
	List<Integer> mutualFeatures = null;
	String epoch = String.valueOf(System.currentTimeMillis()/1000);
	Boolean isWriterInitiated = false;
	PrintWriter writer;
	
	
	public void WriteInComparisonFile(String toWrite, String fileName) throws FileNotFoundException, UnsupportedEncodingException
	{
		
		if(!(new File("Result")).exists()){
			boolean success = (new File("Result")).mkdirs();
		}
		
		
		if(!isWriterInitiated){
			boolean success = (new File("Result/" + epoch)).mkdirs();
		
			
			 writer = new PrintWriter("Result/" + epoch + "/" + fileName  + "_Comparison.txt", "UTF-8");
			 isWriterInitiated = true;
		}
		
		writer.println(toWrite);
		writer.flush();
	}
	
	public void CloseWriter()
	{
		writer.close();	
	}
	
	public void compareFeatures(List<FeatureSelectionResult> r, String fileName) throws FileNotFoundException, UnsupportedEncodingException{
		System.out.println("Starting comparison for "+fileName);
		Iterator<FeatureSelectionResult> it=r.iterator();
		
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
				List<Integer> features=new ArrayList<Integer>();
				for(Integer obj : inst.getFeatures())
					features.add(obj);
				for (int i=0; i<mutualFeatures.size(); i++) {
					Integer e = mutualFeatures.get(i);
					
					if(!features.contains(e)){
						mutualFeatures.remove(e);
						i--;
					}
				}
			}
			
		}
		WriteInComparisonFile("++Compared algorithms:"+algorithms, fileName);
		String listString = mutualFeatures.toString();
		listString = listString.substring(1, listString.length()-1); 
		WriteInComparisonFile("++Mutual features:"+listString, fileName);
		isWriterInitiated = false;
		System.out.println("Comparison done!");
	}

}
