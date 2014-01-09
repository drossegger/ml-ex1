package main;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class CMDReader {
	private Options options;
	private CommandLine cli;
	public CMDReader() {
		Option featureSelectionTechnique = OptionBuilder.withArgName("technique").hasArg()
				.withDescription("use specified feature selection technique").create("fst");
		Option parameters = OptionBuilder.withArgName("parameter=value").hasArgs(2).withValueSeparator()
				.withDescription("use value for given parameter").create("p");

		options=new Options();
		options.addOption(featureSelectionTechnique);
		options.addOption(parameters);
	}

	public void parse(String[] args) {
		CommandLineParser parser=new GnuParser();
		try{
			cli=parser.parse(options,args);
		}
		catch(ParseException e){
			System.out.println("Error parsing command line options: "+e.getMessage());
		}
	}
	public boolean useFeature(){
		return cli.hasOption("fst");
	}
	public String getFeature(){
		return cli.getOptionValue("fst");
	}
	public String getParameters(){
		return "";
	}
}
;