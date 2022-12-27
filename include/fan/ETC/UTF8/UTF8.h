#pragma once

#include _WITCH_PATH(WITCH.h)

uint8_t UTF8_SizeOfCharacter(uint8_t byte){
	if(byte < 0x80){
		return 1;
	}
	if(byte < 0xc0){
		/* error */
		return 1;
	}
	if(byte < 0xe0){
		return 2;
	}
	if(byte < 0xf0){
		return 3;
	}
	if(byte <= 0xf7){
		return 4;
	}
	/* error */
	return 1;
}

bool UTF8_IsCharacterValid(uint8_t byte){
	if(byte < 0x80){
		return 1;
	}
	if(byte < 0xc0){
		return 0;
	}
	if(byte < 0xe0){
		return 1;
	}
	if(byte < 0xf0){
		return 1;
	}
	if(byte <= 0xf7){
		return 1;
	}
	return 0;
}
