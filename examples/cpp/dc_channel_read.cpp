#define _INCLUDE_TOKEN(p0, p1) <p0/p1>


#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(system.h)

#define loco_window
#define loco_context
#include _FAN_PATH(graphics/loco.h)

#include <windows.h>
#include <winhttp.h>
#include <iostream>

#pragma comment(lib, "winhttp.lib")

int main() {
  HINTERNET hSession = WinHttpOpen(L"WinHTTP Example/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!hSession) {
    std::cerr << "WinHttpOpen failed: " << GetLastError() << std::endl;
    return -1;
  }

  HINTERNET hConnect = WinHttpConnect(hSession, L"discord.com", INTERNET_DEFAULT_HTTPS_PORT, 0);
  if (!hConnect) {
    std::cerr << "WinHttpConnect failed: " << GetLastError() << std::endl;
    WinHttpCloseHandle(hSession);
    return -1;
  }

  HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", L"/api/v9/channels/1140364831057260584/messages", NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
  if (!hRequest) {
    std::cerr << "WinHttpOpenRequest failed: " << GetLastError() << std::endl;
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return -1;
  }

  LPCWSTR additionalHeaders = L"Content-Type: application/x-www-form-urlencoded\r\nauthorization: token\r\n";
  if (!WinHttpAddRequestHeaders(hRequest, additionalHeaders, -1L, WINHTTP_ADDREQ_FLAG_ADD)) {
    std::cerr << "WinHttpAddRequestHeaders failed: " << GetLastError() << std::endl;
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return -1;
  }

  if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
    std::cerr << "WinHttpSendRequest failed: " << GetLastError() << std::endl;
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return -1;
  }

  if (!WinHttpReceiveResponse(hRequest, NULL)) {
    std::cerr << "WinHttpReceiveResponse failed: " << GetLastError() << std::endl;
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return -1;
  }

  DWORD statusCode = 0;
  DWORD statusCodeSize = sizeof(statusCode);
  if (!WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &statusCodeSize, WINHTTP_NO_HEADER_INDEX)) {
    std::cerr << "WinHttpQueryHeaders failed: " << GetLastError() << std::endl;
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return -1;
  }
  std::cout << "Status code: " << statusCode << std::endl;

  DWORD availableData = 0;
  while (WinHttpQueryDataAvailable(hRequest, &availableData)) {
    char* buffer = new char[availableData + 1];
    ZeroMemory(buffer, availableData + 1);

    DWORD bytesRead = 0;
    if (!WinHttpReadData(hRequest, buffer, availableData, &bytesRead)) {
      std::cerr << "WinHttpReadData failed: " << GetLastError() << std::endl;
      delete[] buffer;
      break;
    }

    fan::string text(buffer, buffer + bytesRead);
    uint64_t offset = 0;
    while (1) {
      auto found = text.find("content\":", offset);
      if (found == std::string::npos) {
        break;
      }
      auto old_p = found;
      found = text.find(",", old_p);
      if (found == std::string::npos) {
        break;
      }
      fan::print(text.substr(old_p, found - old_p));
      offset += found;
    }

    //std::cout.write(buffer, bytesRead);

    delete[] buffer;

    break;
  }

  WinHttpCloseHandle(hRequest);
  WinHttpCloseHandle(hConnect);
  WinHttpCloseHandle(hSession);

  return 0;
}
