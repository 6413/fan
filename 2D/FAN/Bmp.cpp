#include "FAN/bmp.hpp"

#include <iostream>
#include <fstream>

#include <stdint.h>

namespace BMP_Offsets {
	constexpr::ptrdiff_t PIXELDATA = 0xA;
	constexpr::ptrdiff_t WIDTH = 0x12;
	constexpr::ptrdiff_t HEIGHT = 0x16;
}

char* LoadBMP(const char* path, Object& object) {
	FILE* file = fopen(path, "rb");
	if (!file) {
		printf("wrong path %s", path);
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	size_t size = ftell(file);
	fseek(file, 0, SEEK_SET);
	char* data = (char*)malloc(size);
	fread(data, 1, size, file);
	fclose(file);

	uint32_t pixelOffset =  *(uint32_t*)(data + BMP_Offsets::PIXELDATA);
	object.width =  *(uint32_t*)(data + BMP_Offsets::WIDTH);
	object.height = *(uint32_t*)(data + BMP_Offsets::HEIGHT);

	return data + pixelOffset;
}