package input;
import java.io.File;
import java.io.FilenameFilter;
import java.util.LinkedList;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class InstanceReader {
	private final File path;
	
	public List <String> instaceAddresses = new LinkedList <String>();

	public InstanceReader(String path) {
		this.path = new File(path);
	}

	public List<Instances> read() throws Exception {
		List<Instances> l = new LinkedList<Instances>();
		for (final File file : path.listFiles(new FilenameFilter() {
			@Override
			public boolean accept(File dir, String name) {
				return name.toLowerCase().endsWith(".csv")
						|| name.toLowerCase().endsWith(".arff");
			}
		})) {
			DataSource source = new DataSource(file.getAbsolutePath());
			l.add(source.getDataSet());
			instaceAddresses.add(file.getName());

		}
		return l;

	}

}
