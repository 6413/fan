#include <fan/pch.h>
#include <Windows.h>
#include <mmsystem.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <time.h>
#include <iostream>

using namespace std;

#pragma comment(lib, "Winmm.lib")

CHAR fileName[] = "loopback-capture.wav";
BOOL bDone = FALSE;
HMMIO hFile = NULL;

// REFERENCE_TIME time units per second and per millisecond
#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  100000

#define EXIT_ON_ERROR(hres)  \
                  if (FAILED(hres)) { goto Exit; }
#define SAFE_RELEASE(punk)  \
                  if ((punk) != NULL)  \
                    { (punk)->Release(); (punk) = NULL; }

const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);

class MyAudioSink
{
public:
  HRESULT CopyData(BYTE* pData, UINT32 NumFrames, BOOL* pDone, WAVEFORMATEX* pwfx, HMMIO hFile);
};

HRESULT WriteWaveHeader(HMMIO hFile, LPCWAVEFORMATEX pwfx, MMCKINFO* pckRIFF, MMCKINFO* pckData);
HRESULT FinishWaveFile(HMMIO hFile, MMCKINFO* pckRIFF, MMCKINFO* pckData);
HRESULT RecordAudioStream(MyAudioSink* pMySink);

int main()
{
  clock();

  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);

  // Create file
  MMIOINFO mi = { 0 };
  hFile = mmioOpen(
    // some flags cause mmioOpen write to this buffer
    // but not any that we're using
    fileName,
    &mi,
    MMIO_WRITE | MMIO_CREATE
  );

  if (NULL == hFile) {
    wprintf(L"mmioOpen(\"%ls\", ...) failed. wErrorRet == %u", fileName, GetLastError());
    return E_FAIL;
  }

  MyAudioSink AudioSink;
  RecordAudioStream(&AudioSink);

  mmioClose(hFile, 0);

  CoUninitialize();
  return 0;
}


HRESULT MyAudioSink::CopyData(BYTE* pData, UINT32 NumFrames, BOOL* pDone, WAVEFORMATEX* pwfx, HMMIO hFile)
{
  HRESULT hr = S_OK;

  if (0 == NumFrames) {
    wprintf(L"IAudioCaptureClient::GetBuffer said to read 0 frames\n");
    return E_UNEXPECTED;
  }

  LONG lBytesToWrite = NumFrames * pwfx->nBlockAlign;
#pragma prefast(suppress: __WARNING_INCORRECT_ANNOTATION, "IAudioCaptureClient::GetBuffer SAL annotation implies a 1-byte buffer")
  LONG lBytesWritten = mmioWrite(hFile, reinterpret_cast<PCHAR>(pData), lBytesToWrite);
  if (lBytesToWrite != lBytesWritten) {
    wprintf(L"mmioWrite wrote %u bytes : expected %u bytes", lBytesWritten, lBytesToWrite);
    return E_UNEXPECTED;
  }

  static int CallCount = 0;
  cout << "CallCount = " << CallCount++ << "NumFrames: " << NumFrames << endl;

  if (clock() > 10 * CLOCKS_PER_SEC) //Record 10 seconds. From the first time call clock() at the beginning of the main().
    *pDone = true;

  return S_OK;
}

HRESULT RecordAudioStream(MyAudioSink* pMySink)
{
  HRESULT hr;
  REFERENCE_TIME hnsRequestedDuration = REFTIMES_PER_SEC;
  REFERENCE_TIME hnsActualDuration;
  UINT32 bufferFrameCount;
  UINT32 numFramesAvailable;
  IMMDeviceEnumerator* pEnumerator = NULL;
  IMMDevice* pDevice = NULL;
  IAudioClient* pAudioClient = NULL;
  IAudioCaptureClient* pCaptureClient = NULL;
  WAVEFORMATEX* pwfx = NULL;
  UINT32 packetLength = 0;

  BYTE* pData;
  DWORD flags;

  MMCKINFO ckRIFF = { 0 };
  MMCKINFO ckData = { 0 };

  hr = CoCreateInstance(
    CLSID_MMDeviceEnumerator, NULL,
    CLSCTX_ALL, IID_IMMDeviceEnumerator,
    (void**)&pEnumerator);
  EXIT_ON_ERROR(hr)

    hr = pEnumerator->GetDefaultAudioEndpoint(
      eRender, eConsole, &pDevice);
  EXIT_ON_ERROR(hr)

    hr = pDevice->Activate(
      IID_IAudioClient, CLSCTX_ALL,
      NULL, (void**)&pAudioClient);
  EXIT_ON_ERROR(hr)

    hr = pAudioClient->GetMixFormat(&pwfx);
  EXIT_ON_ERROR(hr)

    hr = pAudioClient->Initialize(
      AUDCLNT_SHAREMODE_SHARED,
      AUDCLNT_STREAMFLAGS_LOOPBACK,
      hnsRequestedDuration,
      0,
      pwfx,
      NULL);
  EXIT_ON_ERROR(hr)

    // Get the size of the allocated buffer.
    hr = pAudioClient->GetBufferSize(&bufferFrameCount);
  EXIT_ON_ERROR(hr)

    hr = pAudioClient->GetService(
      IID_IAudioCaptureClient,
      (void**)&pCaptureClient);
  EXIT_ON_ERROR(hr)

    hr = WriteWaveHeader((HMMIO)hFile, pwfx, &ckRIFF, &ckData);
  if (FAILED(hr)) {
    // WriteWaveHeader does its own logging
    return hr;
  }

  // Calculate the actual duration of the allocated buffer.
  hnsActualDuration = (double)REFTIMES_PER_SEC *
    bufferFrameCount / pwfx->nSamplesPerSec;

  hr = pAudioClient->Start();  // Start recording.
  EXIT_ON_ERROR(hr)

    // Each loop fills about half of the shared buffer.
    while (bDone == FALSE)
    {
      // Sleep for half the buffer duration.
      Sleep(hnsActualDuration / REFTIMES_PER_MILLISEC / 2);

      hr = pCaptureClient->GetNextPacketSize(&packetLength);
      EXIT_ON_ERROR(hr)

        while (packetLength != 0)
        {
          // Get the available data in the shared buffer.
          hr = pCaptureClient->GetBuffer(
            &pData,
            &numFramesAvailable,
            &flags, NULL, NULL);
          EXIT_ON_ERROR(hr)

            if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
            {
              pData = NULL;  // Tell CopyData to write silence.
            }

          // Copy the available capture data to the audio sink.
          hr = pMySink->CopyData(
            pData, numFramesAvailable, &bDone, pwfx, (HMMIO)hFile);
          EXIT_ON_ERROR(hr)

            hr = pCaptureClient->ReleaseBuffer(numFramesAvailable);
          EXIT_ON_ERROR(hr)

            hr = pCaptureClient->GetNextPacketSize(&packetLength);
          EXIT_ON_ERROR(hr)
        }
    }

  hr = pAudioClient->Stop();  // Stop recording.
  EXIT_ON_ERROR(hr)

    hr = FinishWaveFile((HMMIO)hFile, &ckData, &ckRIFF);
  if (FAILED(hr)) {
    // FinishWaveFile does it's own logging
    return hr;
  }

Exit:
  CoTaskMemFree(pwfx);
  SAFE_RELEASE(pEnumerator)
    SAFE_RELEASE(pDevice)
    SAFE_RELEASE(pAudioClient)
    SAFE_RELEASE(pCaptureClient)

    return hr;
}

HRESULT WriteWaveHeader(HMMIO hFile, LPCWAVEFORMATEX pwfx, MMCKINFO* pckRIFF, MMCKINFO* pckData) {
  MMRESULT result;

  // make a RIFF/WAVE chunk
  pckRIFF->ckid = MAKEFOURCC('R', 'I', 'F', 'F');
  pckRIFF->fccType = MAKEFOURCC('W', 'A', 'V', 'E');

  result = mmioCreateChunk(hFile, pckRIFF, MMIO_CREATERIFF);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioCreateChunk(\"RIFF/WAVE\") failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  // make a 'fmt ' chunk (within the RIFF/WAVE chunk)
  MMCKINFO chunk;
  chunk.ckid = MAKEFOURCC('f', 'm', 't', ' ');
  result = mmioCreateChunk(hFile, &chunk, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioCreateChunk(\"fmt \") failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  // write the WAVEFORMATEX data to it
  LONG lBytesInWfx = sizeof(WAVEFORMATEX) + pwfx->cbSize;
  LONG lBytesWritten =
    mmioWrite(
      hFile,
      reinterpret_cast<PCHAR>(const_cast<LPWAVEFORMATEX>(pwfx)),
      lBytesInWfx
    );
  if (lBytesWritten != lBytesInWfx) {
    wprintf(L"mmioWrite(fmt data) wrote %u bytes; expected %u bytes", lBytesWritten, lBytesInWfx);
    return E_FAIL;
  }

  // ascend from the 'fmt ' chunk
  result = mmioAscend(hFile, &chunk, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioAscend(\"fmt \" failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  // make a 'fact' chunk whose data is (DWORD)0
  chunk.ckid = MAKEFOURCC('f', 'a', 'c', 't');
  result = mmioCreateChunk(hFile, &chunk, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioCreateChunk(\"fmt \") failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  // write (DWORD)0 to it
  // this is cleaned up later
  DWORD frames = 0;
  lBytesWritten = mmioWrite(hFile, reinterpret_cast<PCHAR>(&frames), sizeof(frames));
  if (lBytesWritten != sizeof(frames)) {
    wprintf(L"mmioWrite(fact data) wrote %u bytes; expected %u bytes", lBytesWritten, (UINT32)sizeof(frames));
    return E_FAIL;
  }

  // ascend from the 'fact' chunk
  result = mmioAscend(hFile, &chunk, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioAscend(\"fact\" failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  // make a 'data' chunk and leave the data pointer there
  pckData->ckid = MAKEFOURCC('d', 'a', 't', 'a');
  result = mmioCreateChunk(hFile, pckData, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioCreateChunk(\"data\") failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  return S_OK;
}

HRESULT FinishWaveFile(HMMIO hFile, MMCKINFO* pckRIFF, MMCKINFO* pckData) {
  MMRESULT result;

  result = mmioAscend(hFile, pckData, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioAscend(\"data\" failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  result = mmioAscend(hFile, pckRIFF, 0);
  if (MMSYSERR_NOERROR != result) {
    wprintf(L"mmioAscend(\"RIFF/WAVE\" failed: MMRESULT = 0x%08x", result);
    return E_FAIL;
  }

  return S_OK;
}