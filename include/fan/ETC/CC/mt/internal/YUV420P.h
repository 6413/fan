#include _WITCH_PATH(ETC/CC/internal/YUV420P.h)

bool _CC_mt_f_YUV420P_RGB24_thread_in(EV_t *listener, _CC_mt_tp_t *arg){
	_CC_YUV420P_RGB24_2x2(arg->width, arg->height, arg->sstride, arg->dstride, arg->sdata, arg->ddata);
	_CC_mt_thread_end(listener, arg);
	return CC_mt_set_active_tp_delete;
}

bool _CC_mt_f_YUV420P_RGB24(
	CC_mt_t *ccmt,
	EV_t *listener,
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

	_CC_mt_open_2_1x2x2_1(ccmt, listener, width, height, sstride, dstride, sdata, ddata, (EV_tp_cb_outside_t)_CC_mt_f_YUV420P_RGB24_thread_in);

	return 0;
}

_CC_mt_convert_func_t _CC_mt_YUV420P_getfmt(uint32_t dfmt){
	switch(dfmt){
		case ETC_PIXF_RGB24:{
			return _CC_mt_f_YUV420P_RGB24;
		}
		default:{
			return 0;
		}
	}
}
