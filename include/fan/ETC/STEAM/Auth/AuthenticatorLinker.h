#pragma once

#include _WITCH_PATH(WITCH.h)

#include _WITCH_PATH(ETC/STEAM/Auth/SteamGuardAccount.h)

typedef struct{
	uint8_t phonenum[16];
	uint8_t deviceid[64];
}STEAM_AUTH_linker_t;
