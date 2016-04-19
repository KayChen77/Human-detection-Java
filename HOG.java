package ProjectTwo;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

import javax.imageio.ImageIO;

public class HOG {
	private BufferedImage Image;
	public HOG(BufferedImage img){
		Image = img;
	}
	
    //convert the color image into grayscale image,
	public int[][] gray(BufferedImage image){
		int[][] array = new int[image.getHeight()-32][image.getWidth()-32];
        for (int rPixel = 16; rPixel < image.getHeight() - 16; rPixel++)
     	{
    	 	for (int cPixel = 16; cPixel < image.getWidth() - 16; cPixel++)
	 		{				 
				array[rPixel-16][cPixel-16] = (int)Math.round((0.21)*(image.getRGB(cPixel,rPixel) >> 16 & 0xFF) + 0.72*(image.getRGB(cPixel,rPixel) >> 8 & 0xFF)+(0.07)*(image.getRGB(cPixel,rPixel) & 0xFF)); 
	 		}
     	}
        return array;
	}	

	//compute the gradient in vertical direction
	public  int[][] rgradient(int[][] array){
        int[][] rmagarray = new int[array.length][array[0].length];
        for (int rPixel = 1; rPixel < array.length - 1; rPixel++)
     	{
    	 	for (int cPixel = 0; cPixel < array[0].length; cPixel++)
	 		{
				rmagarray[rPixel][cPixel] = (array[rPixel+1][cPixel] - array[rPixel-1][cPixel]); 
	 		}
        }
        return rmagarray;
	}
	
	//compute the gradient in horizontal direction
	public  int[][] cgradient(int[][] array){
        int[][] cmagarray = new int[array.length][array[0].length];
         for (int rPixel = 0; rPixel < array.length; rPixel++)
         	{
        	 	for (int cPixel = 1; cPixel < array[0].length - 1; cPixel++)
        	 		{
						cmagarray[rPixel][cPixel] = (array[rPixel][cPixel+1] - array[rPixel][cPixel-1]);
        	 		}
            }
         return cmagarray;
	}
	
	//compute the normalized gradient magnitude for each pixel
	public  int[][] magnitude(int[][] rarray,int[][] carray){	
		int[][] magnitude2D = new int[carray.length][rarray[0].length];
		for (int rPixel = 1; rPixel < carray.length-1; rPixel++)
     	{
    	 	for (int cPixel = 1; cPixel < carray[1].length-1; cPixel++)
    	 	{
		     magnitude2D[rPixel][cPixel] = (int)Math.round(Math.pow((Math.pow(carray[rPixel][cPixel], 2)+ Math.pow(rarray[rPixel][cPixel], 2)),0.5)/Math.pow(2, 0.5));   
    	 	}
	    }
		
		//undefined pixels
    	 for (int rPixel = 0; rPixel < carray.length; rPixel++)
 	 	{
    		    magnitude2D[rPixel][0] = 0;
    		    magnitude2D[rPixel][carray[1].length-1] = 0;
 	 	}
     	 for (int cPixel = 0; cPixel < rarray[0].length; cPixel++)
  	 	{
     		 	magnitude2D[0][cPixel] = 0;
     		 	magnitude2D[carray.length-1][cPixel] = 0;
  	 	}
    	return magnitude2D;
     }
	
	//use arctan to compute the gradient angle
	public  double[][] angle(int[][] rarray,int[][] carray){
		double angleArray[][] = new double [carray.length][rarray[1].length];
		 for(int rPixel = 1; rPixel < carray.length-1; rPixel++)
		    {
		        for(int cPixel = 1;cPixel < carray[1].length-1; cPixel++)
		        {
		        	if (carray[rPixel][cPixel] == 0 && rarray[rPixel][cPixel] != 0){
		        		angleArray[rPixel][cPixel] = -90.0;
		        	}
		        	if (carray[rPixel][cPixel] == 0 && rarray[rPixel][cPixel] == 0){ 
		        		angleArray[rPixel][cPixel] = 180;
		        		//the range of arctan is (-90, 90), so normal result can't be 180 and this represents undefined.
		        	}
		        	if (carray[rPixel][cPixel] != 0){
		        	angleArray[rPixel][cPixel] = -Math.toDegrees(Math.atan(((double)(rarray[rPixel][cPixel]))/((double)(carray[rPixel][cPixel]))));
		        	}
		        } 
		    }		 
		 return angleArray;
	}
	
	//quantize the gradient angle into 9 bins
		public  int[][] quantizedArray(double angle[][]){
			int quantized[][] = new int [angle.length][angle[1].length];
			for(int rPixel = 1; rPixel < angle.length-1; rPixel++)
		    {
		        for(int cPixel = 1;cPixel < angle[1].length-1; cPixel++)
		        {
		        	if ( angle[rPixel][cPixel] < 20.0 && angle[rPixel][cPixel] >= 0){
		        		quantized[rPixel][cPixel] = 1;
		        	}
		        	if ( angle[rPixel][cPixel] < 40.0 && angle[rPixel][cPixel] >= 20.0){
		        		quantized[rPixel][cPixel] = 2;
		        	}
		        	if ( angle[rPixel][cPixel] < 60.0 && angle[rPixel][cPixel] >= 40.0){
		        		quantized[rPixel][cPixel] = 3;
		        	}
		        	if ( angle[rPixel][cPixel] < 80.0 && angle[rPixel][cPixel] >= 60.0){
		        		quantized[rPixel][cPixel] = 4;
		        	}
		        	if ( angle[rPixel][cPixel] < 100.0 && angle[rPixel][cPixel] >= 80.0){
		        		quantized[rPixel][cPixel] = 5;
		        	}
		        	if ( angle[rPixel][cPixel] < -80.0 && angle[rPixel][cPixel] >= -100.0){
		        		quantized[rPixel][cPixel] = 5;
		        	}
		        	if ( angle[rPixel][cPixel] < -60.0 && angle[rPixel][cPixel] >= -80.0){
		        		quantized[rPixel][cPixel] = 6;
		        	}
		        	if ( angle[rPixel][cPixel] < -40.0 && angle[rPixel][cPixel] >= -60.0){
		        		quantized[rPixel][cPixel] = 7;
		        	}
		        	if ( angle[rPixel][cPixel] < -20.0 && angle[rPixel][cPixel] >= -40.0){
		        		quantized[rPixel][cPixel] = 8;
		        	}
		        	if ( angle[rPixel][cPixel] < 0 && angle[rPixel][cPixel] >= -20.0){
		        		quantized[rPixel][cPixel] = 9;
		        	}
		        	if ( angle[rPixel][cPixel] == 180){
			        	quantized[rPixel][cPixel] = -1;
			        	// both Ix and Iy are 0
		        	}
		        }
		    }
			//undefined pixels
			for(int rPixel = 0; rPixel < angle.length; rPixel++){
	        	quantized[rPixel][0] = -1;
	        	quantized[rPixel][angle[1].length-1] = -1;
	        }
	        for(int cPixel = 0; cPixel < angle[1].length; cPixel++){
	        	quantized[0][cPixel] = -1;
	        	quantized[angle.length-1][cPixel] = -1;
	        }
			return quantized;
		}

	
	//compute the histogram for each cell
		public  double[][] cellhist(double[][] angleArray, int[][] quantizedAngleArray, int[][] magArray) throws FileNotFoundException{
			int m = quantizedAngleArray.length / 8;
			int n = quantizedAngleArray[1].length / 8;
			int number = m*n;  //the total number of cells
			double histogram[][] = new double [number][9];
			int row = 0;

			//compute the histogram of each cell
			for (int i = 1; i <= m; i++ ){
				for (int j = 1; j <= n; j++ ){
					for (int x = 8*i - 8; x <= 8*i-1; x++){
						for(int y = 8*j - 8; y <= 8*j-1; y++){
							switch (quantizedAngleArray[x][y]){
							case 1:
								if(angleArray[x][y] < 10.0){
								histogram[row][0] = histogram[row][0] + (double)magArray[x][y]*((angleArray[x][y] + 10.0)/20.0);
								histogram[row][8] = histogram[row][8] + (double)magArray[x][y]*((10.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= 10.0){
								histogram[row][0] = histogram[row][0] + (double)magArray[x][y]*((30.0 - angleArray[x][y])/20.0);
								histogram[row][1] = histogram[row][1] + (double)magArray[x][y]*((angleArray[x][y] - 10.0)/20.0);
								}
								break;
							case 2:
								if(angleArray[x][y] < 30.0){
								histogram[row][1] = histogram[row][1] + (double)magArray[x][y]*((angleArray[x][y] - 10.0)/20.0);
								histogram[row][0] = histogram[row][0] + (double)magArray[x][y]*((30.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= 30.0){
								histogram[row][1] = histogram[row][1] + (double)magArray[x][y]*((50.0 - angleArray[x][y])/20.0);
								histogram[row][2] = histogram[row][2] + (double)magArray[x][y]*((angleArray[x][y] - 30.0)/20.0);
								}
								break;
							case 3:
								if(angleArray[x][y] < 50.0){
								histogram[row][2] = histogram[row][2] + (double)magArray[x][y]*((angleArray[x][y] - 30.0)/20.0);
								histogram[row][1] = histogram[row][1] + (double)magArray[x][y]*((50.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= 50.0){
								histogram[row][2] = histogram[row][2] + (double)magArray[x][y]*((70.0 - angleArray[x][y])/20.0);
								histogram[row][3] = histogram[row][3] + (double)magArray[x][y]*((angleArray[x][y] - 50.0)/20.0);
								}
								break;
							case 4:
								if(angleArray[x][y] < 70.0){
								histogram[row][3] = histogram[row][3] + (double)magArray[x][y]*((angleArray[x][y] - 50.0)/20.0);
								histogram[row][2] = histogram[row][2] + (double)magArray[x][y]*((70.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= 70.0){
								histogram[row][3] = histogram[row][3] + (double)magArray[x][y]*((90.0 - angleArray[x][y])/20.0);
								histogram[row][4] = histogram[row][4] + (double)magArray[x][y]*((angleArray[x][y] - 70.0)/20.0);
								}
								break;
							case 5:
								if(angleArray[x][y] < -80.0){
								histogram[row][4] = histogram[row][4] + (double)magArray[x][y]*((-70.0 - angleArray[x][y])/20.0);
								histogram[row][5] = histogram[row][5] + (double)magArray[x][y]*((angleArray[x][y] + 90.0)/20.0);
								}
								if(angleArray[x][y] >= 80.0){
								histogram[row][4] = histogram[row][4] + (double)magArray[x][y]*((angleArray[x][y] - 70.0)/20.0);
								histogram[row][3] = histogram[row][3] + (double)magArray[x][y]*((90.0 - angleArray[x][y])/20.0);
								}
								break;
							case 6:
								if(angleArray[x][y] < -70.0){
								histogram[row][5] = histogram[row][5] + (double)magArray[x][y]*((angleArray[x][y] + 90.0)/20.0);
								histogram[row][4] = histogram[row][4] + (double)magArray[x][y]*((-70.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= -70.0){
								histogram[row][5] = histogram[row][5] + (double)magArray[x][y]*((-50.0 - angleArray[x][y])/20.0);
								histogram[row][6] = histogram[row][6] + (double)magArray[x][y]*((angleArray[x][y] + 70.0)/20.0);
								}
								break;
							case 7:
								if(angleArray[x][y] < -50.0){
								histogram[row][6] = histogram[row][6] + (double)magArray[x][y]*((angleArray[x][y] + 70.0)/20.0);
								histogram[row][5] = histogram[row][5] + (double)magArray[x][y]*((-50.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= -50.0){
								histogram[row][6] = histogram[row][6] + (double)magArray[x][y]*((-30.0 - angleArray[x][y])/20.0);
								histogram[row][7] = histogram[row][7] + (double)magArray[x][y]*((angleArray[x][y] + 50.0)/20.0);
								}
								break;	
							case 8:
								if(angleArray[x][y] < -30.0){
								histogram[row][7] = histogram[row][7] + (double)magArray[x][y]*((angleArray[x][y] + 50.0)/20.0);
								histogram[row][6] = histogram[row][6] + (double)magArray[x][y]*((-30.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= -30.0){
								histogram[row][7] = histogram[row][7] + (double)magArray[x][y]*((-10.0 - angleArray[x][y])/20.0);
								histogram[row][8] = histogram[row][8] + (double)magArray[x][y]*((angleArray[x][y] + 30.0)/20.0);
								}
								break;
							case 9:
								if(angleArray[x][y] < -10.0){
								histogram[row][8] = histogram[row][8] + (double)magArray[x][y]*((angleArray[x][y] + 30.0)/20.0);
								histogram[row][7] = histogram[row][7] + (double)magArray[x][y]*((-10.0 - angleArray[x][y])/20.0);
								}
								if(angleArray[x][y] >= -10.0){
								histogram[row][8] = histogram[row][8] + (double)magArray[x][y]*((10.0 - angleArray[x][y])/20.0);
								histogram[row][0] = histogram[row][0] + (double)magArray[x][y]*((angleArray[x][y] + 10.0)/20.0);
								}
								break;								
							}
							
						}
					}
					row++;
				}
			}			
			return histogram;
		}
			
		//construct a block 
   public double[][] block(double[][] cellhist){
	   int row = 0;
	   double[][] blockhist = new double[105][36];
	   double norm[] = new double[105];
	   double sum[] = new double[105];
	   
	   //2*2 cells form a block 
	   for (int i = 0; i < 15; i++){
		   for (int j = 0; j < 7; j++){
				   for (int x = 0; x < 9; x++){  
	        		   blockhist[row][x] = cellhist[8*i+j][x];
	        		   blockhist[row][9+x] = cellhist[8*i+j+1][x];
	        		   blockhist[row][18+x] = cellhist[8*i+j+8][x];
	        		   blockhist[row][27+x] = cellhist[8*i+j+9][x];
	        	   }
	        	  row++; 
	           }		   
	   }
	   
	   //normalize the block histogram
	   for (int i = 0; i < 105; i++){
		   for (int j = 0; j < 36; j++){
			   sum[i] += blockhist[i][j] * blockhist[i][j];	
		   }
	   }	   
	   for (int i = 0; i < 105; i++){
		   norm[i] = Math.pow(sum[i], 0.5);		   
		   for (int j = 0; j < 36; j++){
			   if (norm[i] != 0){
				   blockhist[i][j] = blockhist[i][j] / norm[i];  
			   }
			   else 
				   blockhist[i][j] = 0; 
		   }
	   }
	   return blockhist;
	}
   
   //concatenate 105 blocks' histogram into a final descriptor
   public double[] concate(double blockhist[][]){
	   double[] des = new double[3780];
	   int num = 0;
	   for(int i = 0; i < 105; i++){
		   for (int j = 0; j < 36; j++){
			   des[num] = blockhist[i][j];
			   num++;
		   }
	   }
	   return des;
   }
   
   public double[] finaldes() throws IOException{
	   int[][] array = gray(Image);
	   int[][] rarray = rgradient(array);
	   int[][] carray = cgradient(array);
	   int[][] mag = magnitude(rarray,carray);
	   double[][] angle = angle(rarray,carray);
	   int[][] quantizedAngle = quantizedArray(angle);
	   double[][] cellhist = cellhist(angle, quantizedAngle, mag);
	   double[][] blockhist = block(cellhist);
	   double[] descriptor = concate(blockhist);
	   return descriptor;
	   
   }
}
