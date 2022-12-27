#pragma once

#include _WITCH_PATH(WITCH.h)

#define JPG_RGB  0
#define JPG_RGBA 1
#define JPG_ARGB 2
#define JPG_BGR  3
#define JPG_BGRA 4
#define JPG_ABGR 5

#define JPG_PERF 1
#define JPG_EFFI 2

#ifndef JPG_set_backend
	#define JPG_set_backend 0
#endif

#if JPG_set_backend == 0
	#include <turbojpeg.h>
#else
	#error ?
#endif

#if JPG_set_backend == 0
	tjhandle _JPG_enhandle, _JPG_dehandle;
	PRE{
		_JPG_enhandle = tjInitCompress();
		_JPG_dehandle = tjInitDecompress();
	}
#endif


int _JPG_defmt(uint32_t fmt){
	switch(fmt){
		#if JPG_set_backend == 0
			case JPG_RGB : return TJPF_RGB;
			case JPG_RGBA: return TJPF_RGBA;
			case JPG_ARGB: return TJPF_ARGB;
			case JPG_BGR : return TJPF_BGR;
			case JPG_BGRA: return TJPF_BGRA;
			case JPG_ABGR: return TJPF_ABGR;
		#endif
		default: return -1;
	}
}

int _JPG_depreset(uint32_t preset){
	switch(preset){
		#if JPG_set_backend == 0
			case 0: return TJFLAG_FASTDCT;
			case JPG_PERF: return TJFLAG_FASTDCT;
			case JPG_EFFI: return TJFLAG_ACCURATEDCT;
		#endif
		default: return -1;
	}
}

uintptr_t JPG_encode(uint8_t *in, uintptr_t x, uintptr_t y, uint32_t infmt, uint8_t **out, f_t quality, uint32_t preset){
	int defmt = _JPG_defmt(infmt);
	int depreset = _JPG_depreset(preset);
	#if JPG_set_backend == 0
		uintptr_t ret = 0;
		if(tjCompress2(_JPG_enhandle, (const unsigned char *)in, x, 0, y, defmt, out, (unsigned long *)&ret, TJSAMP_444, quality * 100, preset) == -1)
			return -1;
		return ret;
	#endif
}

void JPG_out_free(uint8_t *out){
	tjFree(out);
}

bool JPG_info(const uint8_t *in, const uintptr_t size, uintptr_t *x, uintptr_t *y){
	#if JPG_set_backend == 0
		*x = 0;
		*y = 0;
		int ret = tjDecompressHeader(_JPG_dehandle, (unsigned char *)in, size, (int *)x, (int *)y);
		return ret == -1;
	#endif
}

uintptr_t JPG_decode(uint8_t *in, uintptr_t size, uint8_t *out, uint32_t preset){
	#if JPG_set_backend == 0
		uintptr_t x, y;
		int depreset = _JPG_depreset(preset);
		JPG_info(in, size, &x, &y);
		if(tjDecompress2(_JPG_dehandle, in, size, out, x, 0, y, TJPF_RGB, depreset) == -1)
			return -1;
		return x * y * 3;
	#endif
}
