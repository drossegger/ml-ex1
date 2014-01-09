package main;

public class Exercise3 {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		CMDReader cmd=new CMDReader();
		cmd.parse(args);
		if(cmd.useFeature()){
			System.out.println(cmd.getFeature());
		}
		else
			System.out.println("No feature set");
		
	}

}
