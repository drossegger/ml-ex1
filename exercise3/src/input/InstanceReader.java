package input;
import java.io.File;
import java.io.FilenameFilter;
import java.util.LinkedList;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class InstanceReader {
	private final File path;
	
	private List <String> instanceAddresses = new LinkedList <String>();

	public List<String> getInstanceAddresses() {
		return instanceAddresses;
	}

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
			instanceAddresses.add(file.getName());

		}
		return l;

	}

}
