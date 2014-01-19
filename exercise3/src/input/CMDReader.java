package input;

import java.util.HashMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class CMDReader {
	private Options options;
	private CommandLine cli;

	public CMDReader() {
		Option featureSelectionTechnique = OptionBuilder
				.withArgName("technique").hasArg()
				.withDescription("use specified feature selection technique")
				.create("f");
		Option listTechniques = OptionBuilder
				.withDescription("list all available feature selection techniques")
				.create("l");
		/*Option parameters = OptionBuilder.withArgName("parameter=value")
				.hasArgs(2).withValueSeparator()
				.withDescription("use value for given parameter").create("p");*/
		Option instancedir = OptionBuilder
				.withArgName("directory")
				.hasArg()
				.withDescription(
						"directory with instancefiles in csv or arff format")
				.create("d");

		options = new Options();
		options.addOption(featureSelectionTechnique);
		options.addOption(listTechniques);
		options.addOption(instancedir);
	}

	public void parse(String[] args) {
		CommandLineParser parser = new GnuParser();
		try {
			cli = parser.parse(options, args);
		} catch (ParseException e) {
			System.out.println("Error parsing command line options: "
					+ e.getMessage());
		}
	}

	public boolean useFeature() {
		return cli.hasOption("f");
	}

	public boolean isListTechniques(){
		return cli.hasOption("l");
	}
	public String[] getFeatures() {
		return cli.getOptionValues("f");
	}

	public String[] getParameters() {
		return cli.getOptionValues("p");
	}
	public String getInstanceDir(){
		return cli.getOptionValue("d");
	}
	public boolean isDirSet(){
		return cli.hasOption("d");
		
	}
	public void printUsage(){
		HelpFormatter formatter=new HelpFormatter();
		formatter.printHelp("exercise3", options);
	}
	
};
