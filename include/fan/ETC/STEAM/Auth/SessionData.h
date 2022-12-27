#pragma once

#include _WITCH_PATH(WITCH.h)
#include _WITCH_PATH(ETC/HTTP/COOKIE.h)
#include _WITCH_PATH(STR/ttcc.h)

typedef struct{
	uint8_t SessionID[64];
	uint8_t SteamLogin[64];
	uint8_t SteamLoginSecure[64];
	uint8_t WebCookie[64];
	uint8_t OAuthToken[64];
	uint32_t SteamID;
}STEAM_Auth_SessionData_t;

void STEAM_AUTH_SessionData_AddCookies(STEAM_Auth_SessionData_t *SessionData, HTTP_COOKIE_t *cookie){
	uint8_t buf[128];
	STR_ttcc_t ttcc;
	ttcc.ptr = buf;
	ttcc.c = 0;
	ttcc.p = sizeof(buf);

	HTTP_COOKIE_add(cookie, "mobileClientVersion", "0 (2.1.3)", "/", ".steamcommunity.com", 0);
	HTTP_COOKIE_add(cookie, "mobileClient", "android", "/", ".steamcommunity.com", 0);

	STR_FSttcc(&ttcc, "%lu%c", SessionData->SteamID, 0);
	HTTP_COOKIE_add(cookie, "steamid", buf, "/", ".steamcommunity.com", 0);
	HTTP_COOKIE_add(cookie, "steamLogin", SessionData->SteamLogin, "/", ".steamcommunity.com",
		HTTP_COOKIE_HttpOnly_e
	);

	HTTP_COOKIE_add(cookie, "steamLoginSecure", SessionData->SteamLoginSecure, "/", ".steamcommunity.com",
		HTTP_COOKIE_HttpOnly_e | HTTP_COOKIE_Secure_e
	);
	HTTP_COOKIE_add(cookie, "Steam_Language", "english", "/", ".steamcommunity.com", 0);
	HTTP_COOKIE_add(cookie, "dob", "", "/", ".steamcommunity.com", 0);
	HTTP_COOKIE_add(cookie, "sessionid", SessionData->SessionID, "/", ".steamcommunity.com", 0);
}
