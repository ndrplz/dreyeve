#include "eyexgazeacquisition.h"

// static functions, cannot be member of class
void TX_CALLCONVENTION HandleEvent(TX_CONSTHANDLE hAsyncData, TX_USERPARAM userParam);
void TX_CALLCONVENTION OnEngineConnectionStateChanged(TX_CONNECTIONSTATE connectionState, TX_USERPARAM userParam);
void TX_CALLCONVENTION OnSnapshotCommitted(TX_CONSTHANDLE hAsyncData, TX_USERPARAM param);

std::mutex m; // to control access to _gaze
TX_HANDLE g_hGlobalInteractorSnapshot = TX_EMPTY_HANDLE; // Defined

EyeXGazeData::EyeXGazeData(){
	_x = -1;
	_y = -1;
}

EyeXGazeAcquisition::EyeXGazeAcquisition(){
	// initialize and enable the context that is our link to the EyeX Engine.
	_success = txInitializeEyeX(TX_EYEXCOMPONENTOVERRIDEFLAG_NONE, NULL, NULL, NULL, NULL) == TX_RESULT_OK;
	_success &= txCreateContext(&_hContext, TX_FALSE) == TX_RESULT_OK;
	_success &= InitializeGlobalInteractorSnapshot();
	_success &= txRegisterConnectionStateChangedHandler(_hContext, &_hConnectionStateChangedTicket, OnEngineConnectionStateChanged, NULL) == TX_RESULT_OK;
	_success &= txRegisterEventHandler(_hContext, &_hEventHandlerTicket, HandleEvent, this) == TX_RESULT_OK;
	_success &= txEnableConnection(_hContext) == TX_RESULT_OK;

}

/*
* Initializes g_hGlobalInteractorSnapshot with an interactor that has the Gaze Point behavior.
*/
BOOL EyeXGazeAcquisition::InitializeGlobalInteractorSnapshot()
{
	TX_HANDLE hInteractor = TX_EMPTY_HANDLE;
	TX_GAZEPOINTDATAPARAMS params = { TX_GAZEPOINTDATAMODE_LIGHTLYFILTERED };
	BOOL success;

	success = txCreateGlobalInteractorSnapshot(
		_hContext,
		InteractorId,
		&g_hGlobalInteractorSnapshot,
		&hInteractor) == TX_RESULT_OK;
	success &= txCreateGazePointDataBehavior(hInteractor, &params) == TX_RESULT_OK;

	txReleaseObject(&hInteractor);

	return success;
}

/*
* Callback function invoked when an event has been received from the EyeX Engine.
*/
void TX_CALLCONVENTION HandleEvent(TX_CONSTHANDLE hAsyncData, TX_USERPARAM userParam)
{
	TX_HANDLE hEvent = TX_EMPTY_HANDLE;
	TX_HANDLE hBehavior = TX_EMPTY_HANDLE;

	txGetAsyncDataContent(hAsyncData, &hEvent);

	EyeXGazeAcquisition* _obj = static_cast<EyeXGazeAcquisition *>(userParam);

	// NOTE. Uncomment the following line of code to view the event object. The same function can be used with any interaction object.
	//OutputDebugStringA(txDebugObject(hEvent));

	if (txGetEventBehavior(hEvent, &hBehavior, TX_BEHAVIORTYPE_GAZEPOINTDATA) == TX_RESULT_OK) {
		_obj->OnGazeDataEvent(hBehavior);
		txReleaseObject(&hBehavior);
	}

	// NOTE since this is a very simple application with a single interactor and a single data stream, 
	// our event handling code can be very simple too. A more complex application would typically have to 
	// check for multiple behaviors and route events based on interactor IDs.

	txReleaseObject(&hEvent);
}

/*
* Callback function invoked when a snapshot has been committed.
*/
void TX_CALLCONVENTION OnSnapshotCommitted(TX_CONSTHANDLE hAsyncData, TX_USERPARAM param)
{
	// check the result code using an assertion.
	// this will catch validation errors and runtime errors in debug builds. in release builds it won't do anything.

	TX_RESULT result = TX_RESULT_UNKNOWN;
	txGetAsyncDataResultCode(hAsyncData, &result);
	assert(result == TX_RESULT_OK || result == TX_RESULT_CANCELLED);
}

/*
* Callback function invoked when the status of the connection to the EyeX Engine has changed.
*/
void TX_CALLCONVENTION OnEngineConnectionStateChanged(TX_CONNECTIONSTATE connectionState, TX_USERPARAM userParam)
{
	EyeXGazeAcquisition* _obj = static_cast<EyeXGazeAcquisition *>(userParam);

	switch (connectionState) {
	case TX_CONNECTIONSTATE_CONNECTED: {
		BOOL success;
		//printf("The connection state is now CONNECTED (We are connected to the EyeX Engine)\n");
		// commit the snapshot with the global interactor as soon as the connection to the engine is established.
		// (it cannot be done earlier because committing means "send to the engine".)
		success = txCommitSnapshotAsync(g_hGlobalInteractorSnapshot, OnSnapshotCommitted, NULL) == TX_RESULT_OK;
		if (!success) {
			//printf("Failed to initialize the data stream.\n");
		}
		else {
			//printf("Waiting for gaze data to start streaming...\n");
		}
	}
		break;

	case TX_CONNECTIONSTATE_DISCONNECTED:
		//printf("The connection state is now DISCONNECTED (We are disconnected from the EyeX Engine)\n");
		break;

	case TX_CONNECTIONSTATE_TRYINGTOCONNECT:
		//printf("The connection state is now TRYINGTOCONNECT (We are trying to connect to the EyeX Engine)\n");
		break;

	case TX_CONNECTIONSTATE_SERVERVERSIONTOOLOW:
		//printf("The connection state is now SERVER_VERSION_TOO_LOW: this application requires a more recent version of the EyeX Engine to run.\n");
		break;

	case TX_CONNECTIONSTATE_SERVERVERSIONTOOHIGH:
		//printf("The connection state is now SERVER_VERSION_TOO_HIGH: this application requires an older version of the EyeX Engine to run.\n");
		break;
	}
}


/*
* Handles an event from the Gaze Point data stream.
*/
void EyeXGazeAcquisition::OnGazeDataEvent(TX_HANDLE hGazeDataBehavior)
{
	TX_GAZEPOINTDATAEVENTPARAMS eventParams;
	if (txGetGazePointDataEventParams(hGazeDataBehavior, &eventParams) == TX_RESULT_OK) {
		setGazeData(EyeXGazeData(eventParams.X, eventParams.Y));
		//printf("Gaze Data: (%.1f, %.1f) timestamp %.0f ms\n", eventParams.X, eventParams.Y, eventParams.Timestamp);
	}
	else {
		setGazeData(EyeXGazeData(-1, -1));
		//printf("Failed to interpret gaze data event packet.\n");
	}
}


/*
Getters and setters for gaze data
*/
void EyeXGazeAcquisition::setGazeData(EyeXGazeData g){
	m.lock();
	_gaze._x = g._x;
	_gaze._y = g._y;
	m.unlock();
}

EyeXGazeData EyeXGazeAcquisition::getGazeData(){
	m.lock();
	EyeXGazeData ret(_gaze._x, _gaze._y);
	m.unlock();

	return ret;
}


EyeXGazeAcquisition::~EyeXGazeAcquisition(){
	// disable and delete the context.
	txDisableConnection(_hContext);
	txReleaseObject(&g_hGlobalInteractorSnapshot);
	_success = txShutdownContext(_hContext, TX_CLEANUPTIMEOUT_DEFAULT, TX_FALSE) == TX_RESULT_OK;
	_success &= txReleaseContext(&_hContext) == TX_RESULT_OK;
	_success &= txUninitializeEyeX() == TX_RESULT_OK;
}
