#include _WITCH_PATH(ETC/CC/internal/BGRA.h)

bool _CC_st_f_BGRA_YUV420P(
	uint32_t width,
	uint32_t height,
	const uint32_t *sstride,
	const uint32_t *dstride,
	const uint8_t *const *sdata,
	uint8_t *const *ddata
){
	if(EXPECT(width % 2, 0)){
		return 1;
	}
	if(EXPECT(height % 2, 0)){
		return 1;
	}

	_CC_BGRA_YUV420P_2x2(width, height, sstride, dstride, sdata, ddata);

	return 0;
}

bool _CC_st_f_BGRA_NV12(
	uint32_t width,
	uint32_t height,
	const uint32_t *sstride,
	const uint32_t *dstride,
	const uint8_t *const *sdata,
	uint8_t *const *ddata
){
	if(EXPECT(width % 2, 0)){
		return 1;
	}
	if(EXPECT(height % 2, 0)){
		return 1;
	}

	_CC_BGRA_NV12_2x2(width, height, sstride, dstride, sdata, ddata);

	return 0;
}

_CC_st_convert_func_t _CC_st_BGRA_getfmt(uint32_t dfmt){
	switch(dfmt){
		case ETC_PIXF_YUV420p:{
			return _CC_st_f_BGRA_YUV420P;
		}
		case ETC_PIXF_YUVNV12:{
			return _CC_st_f_BGRA_NV12;
		}
		default:{
			return 0;
		}
	}
}
