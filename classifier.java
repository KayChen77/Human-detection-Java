package ProjectTwo;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintStream;

import javax.imageio.ImageIO;

public class classifier {

	public static void main(String[] args) throws IOException{
		String[] posfilenames = readFileNames("pos/");
		String[] negfilenames = readFileNames("neg/");
		String[] testposfilenames = readFileNames("testpos/");
		String[] testnegfilenames = readFileNames("testneg/");
		double[][] posdescriptors = new double[10][3781];
		double[][] negdescriptors = new double[10][3781];

		for (int i = 0; i < 10; i++){
			BufferedImage posimage = readImage(posfilenames[i]);
			BufferedImage negimage = readImage(negfilenames[i]);
			HOG poshogimage =  new HOG(posimage);
			HOG neghogimage =  new HOG(negimage);
			//get image's HOG descriptor
			double[] tmp1 = poshogimage.finaldes();
			double[] tmp2 = neghogimage.finaldes();
			int len = 3780;
			
			//augmented vector
			for (int j = 0; j < len; j++) {
				posdescriptors[i][j] = tmp1[j];
				negdescriptors[i][j] = tmp2[j];		
			}
			posdescriptors[i][len] = 1;
			negdescriptors[i][len] = 1;
			
			//output descriptors of required training images
			if (posfilenames[i].equals("pos/crop001030c.bmp") ||  
				posfilenames[i].equals("pos/crop001034b.bmp")){
				String[] fn = posfilenames[i].split("/");
				hogPrint(fn[1], tmp1);
			}
			if (negfilenames[i].equals("neg/00000003a_cut.bmp")||
				negfilenames[i].equals("neg/00000057a_cut.bmp")){
				String[] fn = negfilenames[i].split("/");
				hogPrint(fn[1], tmp2);
				}
				
		}
        
		double[] posdist = dist(posdescriptors, "pos");
		double[] negdist = dist(negdescriptors, "neg");
		
		//output the distance.
		PrintStream distout = new PrintStream(new FileOutputStream("output/dist.txt"));
		for (int i = 0; i < 10; i++){;
				distout.print(posfilenames[i] + " : " + posdist[i]+"\n"); 
			}			
		for (int i = 0; i < 10; i++){;
				distout.print(negfilenames[i] + " : " + negdist[i]+"\n"); 
			}	
		distout.close();
		
		//train the classifier
		double[] w = train(posdescriptors, negdescriptors);
		
		//output the training order
		PrintStream out = new PrintStream("output/training order.txt");
		for (int i = 0; i < 10; i++){
			out.print(posfilenames[i] + "\n");
		}
		for (int i = 0; i < 10; i++){
			out.print(negfilenames[i] + "\n");
		}
		out.close();
		
		//classify the test images and output the classification results
		PrintStream result = new PrintStream("output/result.txt");
		for (int i = 0; i < 5; i++){
			BufferedImage testposimage = readImage(testposfilenames[i]);
			HOG img = new HOG(testposimage);
			double[] hog = img.finaldes();
			
			//output the descriptors of required test positive image
			if (testposfilenames[i].equals("testpos/crop001008b.bmp")){
				String[] fn = testposfilenames[i].split("/");
				hogPrint(fn[1], hog);
				}
			
			result.println(testposfilenames[i] + " : "+ classify(w, hog));
		}
		for (int i = 0; i < 5; i++){
			BufferedImage testnegimage = readImage(testnegfilenames[i]);
			HOG img = new HOG(testnegimage);
			double[] hog = img.finaldes();
			
			//output the descriptors of required test negative image
			if (testnegfilenames[i].equals("testneg/00000053a_cut.bmp")){
				String[] fn = testnegfilenames[i].split("/");
				hogPrint(fn[1], hog);
			}
			
			result.println(testnegfilenames[i] + " : "+ classify(w, hog));
		}
		result.close();		
		}
	
	public static String[] readFileNames(String folder) throws IOException {
		
		// open the folder you want to read files from
		File dir = new File(folder);
		
		// filter the files you are interested in
		FilenameFilter textFilter = new FilenameFilter() {
			public boolean accept(File dir, String name) {
				String lowercaseName = name.toLowerCase();
				if (lowercaseName.endsWith(".bmp")) {
					return true;
				} else {
					return false;
				}
			}
		};
		
		// returns the files
		File[] listOfFiles = dir.listFiles(textFilter);
		
		// exception case
		if (listOfFiles.length==0) {
			return null;
		}
		
		// stores the file names in an array
		String[] filenames = new String[listOfFiles.length];
		for (int i = 0; i < listOfFiles.length; i++) {
		    if (listOfFiles[i].isFile()) {
		    	filenames[i] = listOfFiles[i].getPath();
		    }
		}		
		return filenames;
	}
	
	public static BufferedImage readImage(String filepath) throws IOException {   
		File input = new File(filepath);
		BufferedImage image = ImageIO.read(input);
		return(image);
	}
	
	//compute the mean descriptors and distance of positive samples and negative samples respectively
	public static double[] dist(double[][] samp, String posneg) throws FileNotFoundException{
		double[] mean = new double[3780];
		double[] dist = new double[10];
		double sum = 0;
		for (int i = 0; i < 3780; i++){
			for (int j = 0; j < 10; j++){
				sum += samp[j][i];
			}
			mean[i] = sum/10.0;
			sum = 0;
		}				
	//output the mean descriptors for positive samples and negative samples respectively	
		hogPrint(posneg+"mean", mean);
		
		for (int i = 0; i < 10; i++){
			for (int j = 0; j < 3780; j++){
				sum += Math.pow((Math.abs(samp[i][j]-mean[j])), 2);
			}
			dist[i] = Math.sqrt(sum);
			sum = 0;
		}
		return dist;
	}

	//train the classifier
	public static double[] train(double[][] possamp, double[][] negsamp) throws FileNotFoundException{
		//output the alpha, w and number of iterations
		PrintStream out = new PrintStream("output/parameter.txt");
		double[] w = new double[3781];
		double sum = 0;
		//int flag = 0;
		boolean flag = true;
		double a = 0.75;// learning rate
		int k = 0;
		int count = 0;
		
		out.println("learning rate alpha = " + a);
		out.println("w:");
		//initial W
		for (int j = 0; j < 3781; j++){
			w[j] = -0.5;
			out.print(w[j] + " ");
			k++;
			if (k == 36){
				out.print("\n");
				k = 0;
			}
		}
		
		// ***start training
		do{
			count++;
			flag = false;
			for (int i = 0; i < 10; i++){
				sum = 0;
				for (int j = 0; j < 3781; j++){
					sum += possamp[i][j] * w[j];
				}
				if (sum <= 0){
					for (int j = 0; j < 3781; j++){
						w[j] += a * possamp[i][j];
					}
					//flag &= ~(1 << i);
					flag = true;
				}
				/*else if (sum > 0){
					//flag |= 1 << i;
				}*/
			}
			
			for (int i = 0; i < 10; i++){
				sum = 0;
				for (int j = 0; j < 3781; j++){
					sum += negsamp[i][j] * w[j];
				}
				if (sum >= 0){
					for (int j = 0; j < 3781; j++){
						w[j] -= a * negsamp[i][j];
					}
					//flag &= ~(1 << 10+i);					
					flag = true;
				}
				/*else if (sum < 0){
					flag |= 1 << 10+i;
				}*/
			}	
		}while (flag != false);
		//} while (flag < 0xfffff);
		//***end training
		
		out.print("\n");
		out.println("number of iterations : " + count);
		out.close();
		return w;
	}
	

	public static String classify(double w[], double[] image) throws IOException{
		double[] testdescriptors = new double[3781];
		double sum = 0;
		int len = 3780;
		String result = "";
		
		//augmented vector
		for (int j = 0; j < len; j++) {
			testdescriptors[j] = image[j];	
		}
			testdescriptors[len] = 1;
			
	    //*** start to classify
		for (int j = 0; j < 3781; j++){
			sum += testdescriptors[j] * w[j];
		}
		if (sum < 0){
			result = "negative";
		}
		else if (sum > 0){
			result = "positive";
		}
		//end classification
		return result;
	}	

	//output descriptors
	public static void hogPrint(String filename, double[] descriptors) throws FileNotFoundException{
		PrintStream out = new PrintStream("output/" + filename + ".txt");
		double tmp;
		int i = 0;
		for (int j = 0; j < 3780; j++){
				tmp = Math.round(descriptors[j]*100) / 100.0;
				out.print(tmp + " ");
				i = i + 1;
				if (i == 36) {
					out.print("\n"); 
				    i = 0;
				}				
		}
		out.close();
	}
}