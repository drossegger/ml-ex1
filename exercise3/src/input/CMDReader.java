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
		Option compareresults = OptionBuilder
				.withDescription(
						"compare results and find mutual features of result files inside a directory")
				.create("c");
		Option help= OptionBuilder
				.withDescription("display this usage information")
				.create("h");
		Option topn= OptionBuilder
				.hasArg()
				.withArgName("n")
				.withDescription("use top <n> attributes")
				.create("n");
		Option attribthresh=OptionBuilder
				.hasArg()
				.withArgName("f")
				.withDescription("consider attributes appearing in f per cent of the result sets")
				.create("t");
		options = new Options();
		options.addOption(help);
		options.addOption(featureSelectionTechnique);
		options.addOption(listTechniques);
		options.addOption(instancedir);
		options.addOption(topn);
		options.addOption(attribthresh);
		options.addOption(compareresults);
	
		
	}

	public void parse(String[] args) {
		CommandLineParser parser = new GnuParser();
		try {
			cli = parser.parse(options, args);
		} catch (ParseException e) {
			System.out.println("Error parsing command line options: ");
			printUsage();
		}
	
	}

	public boolean isAttThresh(){
		return cli.hasOption("t");
	}
	public float getAttThresh(){
		return Float.parseFloat(cli.getOptionValue("t"));
	}
	public boolean isTopN(){
		return cli.hasOption("n");
	}
	public int getTopN(){
		return Integer.parseInt(cli.getOptionValue("n"));
	}
	public boolean isHelp(){
		return cli.hasOption("h");
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
	public boolean compareResult() {
		return cli.hasOption("c");
	}
	public void printUsage(){
		HelpFormatter formatter=new HelpFormatter();
		formatter.printHelp("exercise3", options);
	}
	
};
