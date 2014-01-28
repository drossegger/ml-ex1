package input;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.LinkedList;
import java.util.List;

import model.FeatureSelectionResult;

public class ResultReader {
	public List <String> instaceAddresses = new LinkedList <String>();

	@SuppressWarnings("resource")
	public List<FeatureSelectionResult> read(String path) throws Exception {
		List<FeatureSelectionResult> l = new LinkedList<FeatureSelectionResult>();
		FileReader fr = new FileReader(path);
	    BufferedReader br = new BufferedReader(fr);
	    String s = "";
	    FeatureSelectionResult f = null;
	    
	    while (br.ready()) {
	    	s = br.readLine();
	    	
		    if(s.startsWith("++"))
			{
		    	f = new FeatureSelectionResult();
				f.setName(s.substring(2));
			}
			else if(s.startsWith("--Selected attributes:"))
			{
				String[] items = s.substring("--Selected attributes:".length() + 1).split(",");
				int[] results = new int[items.length];

				for (int i = 0; i < items.length; i++) {
				    results[i] = Integer.parseInt(items[i]);
				}
				f.setFeatures(results);
				
				l.add(f);
			}
	    }
	    return l;
	}

}
