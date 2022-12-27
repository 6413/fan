#include _WITCH_PATH(ETC/CC/internal/YUV420P.h)

bool _CC_st_f_YUV420P_RGB24(
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

	_CC_YUV420P_RGB24_2x2(width, height, sstride, dstride, sdata, ddata);

	return 0;
}

_CC_st_convert_func_t _CC_st_YUV420P_getfmt(uint32_t dfmt){
	switch(dfmt){
		case ETC_PIXF_RGB24:{
			return _CC_st_f_YUV420P_RGB24;
		}
		default:{
			return 0;
		}
	}
}