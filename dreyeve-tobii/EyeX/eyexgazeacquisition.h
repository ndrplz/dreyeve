#pragma once

#include <Windows.h>
#include <stdio.h>
#include <conio.h>
#include <assert.h>
#include <eyex\EyeX.h>
#include <mutex>


extern TX_HANDLE g_hGlobalInteractorSnapshot; // Declared

class EyeXGazeData{
public:
	EyeXGazeData();

	EyeXGazeData(double x, double y){
		_x = x;
		_y = y;
	}

	double _x;
	double _y;
};

class EyeXGazeAcquisition{

	// ID of the global interactor that provides our data stream; must be unique within the application.
	TX_STRING InteractorId = "Twilight Sparkle";

	TX_CONTEXTHANDLE _hContext = TX_EMPTY_HANDLE;
	TX_TICKET _hConnectionStateChangedTicket = TX_INVALID_TICKET;
	TX_TICKET _hEventHandlerTicket = TX_INVALID_TICKET;
	BOOL _success;

	EyeXGazeData _gaze;

	BOOL InitializeGlobalInteractorSnapshot();
	


public:

	EyeXGazeAcquisition();
	~EyeXGazeAcquisition();

	EyeXGazeData getGazeData();
	void setGazeData(EyeXGazeData g);

	void OnGazeDataEvent(TX_HANDLE hGazeDataBehavior);
};