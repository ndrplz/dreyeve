#include <opencv2\opencv.hpp>
#include <string>
#include <fstream>
#include <Windows.h>

#include "EyeX\eyexgazeacquisition.h"

using namespace std;
using namespace cv;


/*	Function to log (x,y) gaze couple on csv	*/
void logGazeDataCsv(ofstream& file, const EyeXGazeData& gaze){
	file << gaze._x << "," << gaze._y << endl;
}

/*	Main function, load a sequence and log gaze	*/
int main(){
	
	string driver_id = "david";	// unique driver identifier

	vector<uchar> sequences = { 1, 6 };	// vector of allowed sequences
	uchar sequence_n = sequences[rand() % sequences.size()];	// dreyeve sequence number

	// create output dir
	string output_dir = "out\\";
	CreateDirectoryA(output_dir.c_str(), NULL);

	// build sequence signature string
	char sequence_str[2];
	sprintf(sequence_str, "%02d", sequence_n);
	string img_folder = "Z:\\DATA\\" + string(sequence_str) + "\\frames\\";

	// initialize video capture object
	VideoCapture cap(string(img_folder) + "%06d.jpg");
	Mat cur_frame;	// frame gathered from capture object

	if (!cap.isOpened())
	{
		cout << "Error: could not open video capture." << endl;
	}

	// initialize EyeX
	EyeXGazeAcquisition tobii;
	string gaze_out_dir = output_dir + driver_id + "\\";
	CreateDirectoryA(gaze_out_dir.c_str(), NULL);
	ofstream csv_file(gaze_out_dir+string(sequence_str)+".csv");

	// loop the sequence
	cvNamedWindow("SEQUENCE", CV_WINDOW_FULLSCREEN); waitKey(); // to fit to screen
	while (cap.read(cur_frame)){
		
		// show the video
		imshow("SEQUENCE", cur_frame);

		// capture and log gaze
		EyeXGazeData gaze = tobii.getGazeData();
		logGazeDataCsv(csv_file, gaze);

		waitKey(1);
	}

	csv_file.close();

	return 0;
}