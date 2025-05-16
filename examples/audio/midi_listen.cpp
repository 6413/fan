
#include <iostream>
#include <windows.h>
#include <mmsystem.h>
#include <vector>
#include <string>
#include <iomanip>
#include <dbt.h>
#include <map>
#include <thread>
#include <mutex>

#pragma comment(lib, "winmm.lib")

// Piano key information structure
struct PianoKey {
  int midiNote;
  std::string noteName;
  bool isPressed;
};

// Global variables
std::vector<PianoKey> piano(88);
std::map<UINT, HMIDIIN> openMidiDevices;
std::mutex deviceMutex;
bool isRunning = true;
HWND hWnd = NULL;

#define WM_MIDIDEVICE_CHANGE (WM_USER + 1)

std::string getNoteNameFromNumber(int midiNote) {
  const char* noteNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
  int octave = (midiNote / 12) - 1;  // MIDI note 60 is C4
  int noteInOctave = midiNote % 12;
  return noteNames[noteInOctave] + std::to_string(octave);
}

void initializePiano() {
  for (int i = 0; i < 88; i++) {
    int midiNote = i + 21;
    piano[i].midiNote = midiNote;
    piano[i].noteName = getNoteNameFromNumber(midiNote);
    piano[i].isPressed = false;
  }
}

void displayPianoState() {
  system("cls");
  std::cout << "88-Key Piano MIDI Monitor\n";
  std::cout << "========================\n";
  std::cout << "Press Ctrl+C to exit\n\n";

  std::lock_guard<std::mutex> lock(deviceMutex);
  std::cout << "Connected MIDI devices: " << openMidiDevices.size() << std::endl;
  for (const auto& device : openMidiDevices) {
    MIDIINCAPS midiInCaps;
    midiInGetDevCaps(device.first, &midiInCaps, sizeof(MIDIINCAPS));
    std::cout << "- [" << device.first << "] " << midiInCaps.szPname << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Currently pressed keys:\n";
  bool anyKeyPressed = false;

  for (const auto& key : piano) {
    if (key.isPressed) {
      anyKeyPressed = true;
      std::cout << std::setw(5) << key.noteName;
    }
  }

  if (!anyKeyPressed) {
    std::cout << "(none)";
  }

  std::cout << "\n\nRecent MIDI activity:\n";
}

void CALLBACK MidiInProc(HMIDIIN hMidiIn, UINT wMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
  if (wMsg == MIM_DATA) {
    BYTE status = (BYTE)(dwParam1 & 0xFF);
    BYTE data1 = (BYTE)((dwParam1 >> 8) & 0xFF);  // Note number
    BYTE data2 = (BYTE)((dwParam1 >> 16) & 0xFF); // Velocity

    BYTE msgType = status & 0xF0;

    if (data1 >= 21 && data1 <= 108) {
      int pianoIndex = data1 - 21;

      if (msgType == 0x90 && data2 > 0) {  // Note On
        piano[pianoIndex].isPressed = true;
        std::cout << "Note On: " << piano[pianoIndex].noteName
          << " (MIDI: " << (int)data1 << ") Velocity: " << (int)data2 << std::endl;
      }
      else if (msgType == 0x80 || (msgType == 0x90 && data2 == 0)) {  // Note Off
        piano[pianoIndex].isPressed = false;
        std::cout << "Note Off: " << piano[pianoIndex].noteName
          << " (MIDI: " << (int)data1 << ")" << std::endl;
      }

      displayPianoState();
    }
  }
  else if (wMsg == MIM_OPEN) {
    std::cout << "MIDI device opened successfully" << std::endl;
  }
  else if (wMsg == MIM_CLOSE) {
    std::cout << "MIDI device closed" << std::endl;
  }
  else if (wMsg == MIM_ERROR) {
    std::cout << "MIDI device error" << std::endl;
  }
}

bool openMidiDevice(UINT deviceID) {
  std::lock_guard<std::mutex> lock(deviceMutex);

  if (openMidiDevices.find(deviceID) != openMidiDevices.end()) {
    return true;
  }

  HMIDIIN hMidiIn = NULL;
  MMRESULT result = midiInOpen(&hMidiIn, deviceID, (DWORD_PTR)(void*)MidiInProc, 0, CALLBACK_FUNCTION);

  if (result != MMSYSERR_NOERROR) {
    std::cout << "Failed to open MIDI device " << deviceID << ". Error code: " << result << std::endl;
    return false;
  }

  result = midiInStart(hMidiIn);
  if (result != MMSYSERR_NOERROR) {
    std::cout << "Failed to start MIDI input for device " << deviceID << ". Error code: " << result << std::endl;
    midiInClose(hMidiIn);
    return false;
  }

  openMidiDevices[deviceID] = hMidiIn;

  MIDIINCAPS midiInCaps;
  midiInGetDevCaps(deviceID, &midiInCaps, sizeof(MIDIINCAPS));
  std::cout << "Successfully connected to MIDI device: " << midiInCaps.szPname << std::endl;

  return true;
}

void closeMidiDevice(UINT deviceID) {
  std::lock_guard<std::mutex> lock(deviceMutex);

  auto it = openMidiDevices.find(deviceID);
  if (it != openMidiDevices.end()) {
    HMIDIIN hMidiIn = it->second;
    midiInStop(hMidiIn);
    midiInClose(hMidiIn);
    openMidiDevices.erase(it);

    std::cout << "Disconnected MIDI device ID: " << deviceID << std::endl;
  }
}

void closeAllMidiDevices() {
  std::lock_guard<std::mutex> lock(deviceMutex);

  for (auto& device : openMidiDevices) {
    HMIDIIN hMidiIn = device.second;
    midiInStop(hMidiIn);
    midiInClose(hMidiIn);
  }

  openMidiDevices.clear();
}

void scanAndOpenMidiDevices() {
  UINT numDevs = midiInGetNumDevs();
  std::cout << "Scanning for MIDI Input Devices: Found " << numDevs << std::endl;

  for (UINT i = 0; i < numDevs; i++) {
    MIDIINCAPS midiInCaps;
    midiInGetDevCaps(i, &midiInCaps, sizeof(MIDIINCAPS));
    std::cout << "[" << i << "] " << midiInCaps.szPname << std::endl;

    // Try to open each device
    openMidiDevice(i);
  }

  if (numDevs == 0) {
    std::cout << "No MIDI devices found. Connect a device and it will be detected automatically." << std::endl;
  }

  displayPianoState();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
  case WM_DEVICECHANGE: {
    // Device change detected
    if (wParam == DBT_DEVICEARRIVAL || wParam == DBT_DEVICEREMOVECOMPLETE) {
      // Trigger device scanning in main thread to avoid thread safety issues
      PostMessage(hwnd, WM_MIDIDEVICE_CHANGE, 0, 0);
    }
    break;
  }
  case WM_DESTROY:
    PostQuitMessage(0);
    break;
  default:
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
  }
  return 0;
}

HWND createHiddenWindow() {
  // Register window class
  WNDCLASSEX wc = { 0 };
  wc.cbSize = sizeof(WNDCLASSEX);
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpszClassName = "MidiDeviceNotification";

  RegisterClassEx(&wc);

  // Create the window
  HWND hwnd = CreateWindowEx(
    0,
    "MidiDeviceNotification",
    "Midi Device Notification Window",
    0,
    0, 0, 0, 0,  // Position and size (not visible)
    HWND_MESSAGE,  // Message-only window
    NULL,
    GetModuleHandle(NULL),
    NULL
  );

  return hwnd;
}

// Register for device notifications
void registerForDeviceNotifications(HWND hwnd) {
  DEV_BROADCAST_DEVICEINTERFACE notificationFilter = { 0 };
  notificationFilter.dbcc_size = sizeof(DEV_BROADCAST_DEVICEINTERFACE);
  notificationFilter.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;

  // Register for device interface changes
  HDEVNOTIFY hDevNotify = RegisterDeviceNotification(
    hwnd,
    &notificationFilter,
    DEVICE_NOTIFY_WINDOW_HANDLE | DEVICE_NOTIFY_ALL_INTERFACE_CLASSES
  );

  if (hDevNotify == NULL) {
    std::cout << "Failed to register for device notifications. Error: " << GetLastError() << std::endl;
  }
}

// Window message processing thread
void messageLoop() {
  MSG msg;
  while (GetMessage(&msg, NULL, 0, 0) > 0) {
    if (msg.message == WM_MIDIDEVICE_CHANGE) {
      // Device change notification
      std::cout << "\nMIDI device change detected - rescanning devices...\n" << std::endl;
      scanAndOpenMidiDevices();
    }
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

// SIGINT handler
BOOL WINAPI ConsoleHandler(DWORD signal) {
  if (signal == CTRL_C_EVENT) {
    isRunning = false;
    PostQuitMessage(0);  // Exit message loop
    return TRUE;
  }
  return FALSE;
}

int main() {
  std::cout << "88-Key Piano MIDI Reader with Auto-Detection\n";
  std::cout << "==========================================\n\n";

  SetConsoleCtrlHandler(ConsoleHandler, TRUE);

  initializePiano();

  hWnd = createHiddenWindow();
  if (hWnd == NULL) {
    std::cout << "Failed to create notification window. Error: " << GetLastError() << std::endl;
    return 1;
  }

  registerForDeviceNotifications(hWnd);

  scanAndOpenMidiDevices();

  std::cout << "\nWaiting for MIDI input from your 88-key piano...\n";
  std::cout << "The program will automatically detect when devices are connected or disconnected.\n";
  std::cout << "Press Ctrl+C to exit.\n\n";

  std::thread msgThread(messageLoop);

  while (isRunning) {
    Sleep(100);
  }

  closeAllMidiDevices();

  if (msgThread.joinable()) {
    msgThread.join();
  }

  std::cout << "Program terminated." << std::endl;
  return 0;
}