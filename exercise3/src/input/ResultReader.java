package input;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.util.LinkedList;
import java.util.List;

import model.FeatureSelectionResult;

public class ResultReader {
	private final File path;
	
	public List <String> instaceAddresses = new LinkedList <String>();

	public ResultReader(String path) {
		this.path = new File(path);
	}

	@SuppressWarnings("resource")
	public List<FeatureSelectionResult> read() throws Exception {
		List<FeatureSelectionResult> l = new LinkedList<FeatureSelectionResult>();
		for (final File file : path.listFiles(new FilenameFilter() {
			@Override
			public boolean accept(File dir, String name) {
				return name.toLowerCase().endsWith(".txt");
			}
		})) {
			if (file.isFile()) {
				FileReader fr = new FileReader(file.getAbsolutePath());
			    BufferedReader br = new BufferedReader(fr);
			    String s = "";
			    FeatureSelectionResult f = new FeatureSelectionResult();
			    
			    while (br.ready()) {
			    	s = br.readLine();
			    	
				    if(s.startsWith("++"))
					{
						f.Name = s.substring(2);
					}
					else if(s.startsWith("--Selected attributes:"))
					{
						String[] items = s.substring("--Selected attributes:".length() + 1).split(",");
						Integer[] results = new Integer[items.length];

						for (int i = 0; i < items.length; i++) {
						    results[i] = Integer.parseInt(items[i]);
						}
						f.Features = results;
					}
			    }
			    
				l.add(f);
		    }
		}
		return l;

	}

}
