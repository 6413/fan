#pragma once

#include _WITCH_PATH(WITCH.h)
#include _WITCH_PATH(MAP/MAP.h)

#include _WITCH_PATH(ETC/STEAM/constants.h)

/*
	dataString = NULL
	cookies = NULL
	headers = NULL
	referer = STEAM_COMMUNITY_BASE
*/
sint32_t STEAM_Auth_SteamWeb_Request0(const void *url, const void *method, const void *dataString, HTTP_COOKIE_t *cookies, MAP_t *headers, const void *referer){
	
}
