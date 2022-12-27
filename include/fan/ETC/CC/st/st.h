#pragma once

#include _WITCH_PATH(WITCH.h)

#include _WITCH_PATH(ETC/PIXF/PIXF.h)

typedef bool (*_CC_st_convert_func_t)(
  uint32_t x,
  uint32_t y,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata
);

#include _WITCH_PATH(ETC/CC/st/internal/RGB24.h)
#include _WITCH_PATH(ETC/CC/st/internal/BGRA.h)
#include _WITCH_PATH(ETC/CC/st/internal/YUV420P.h)

_CC_st_convert_func_t _CC_st_getfmt(uint32_t sfmt, uint32_t dfmt){
  switch(sfmt){
    case ETC_PIXF_RGB24:{
      return _CC_st_RGB24_getfmt(dfmt);
    }
    case ETC_PIXF_BGRA:{
      return _CC_st_BGRA_getfmt(dfmt);
    }
    case ETC_PIXF_YUV420p:{
      return _CC_st_YUV420P_getfmt(dfmt);
    }
    default:{
      return 0;
    }
  }
}

bool CC_st_convert(
  uint32_t sfmt,
  uint32_t dfmt,
  uint32_t x,
  uint32_t y,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata
){
  _CC_st_convert_func_t convert_func = _CC_st_getfmt(sfmt, dfmt);
  if(!convert_func){
    return 1;
  }
  return convert_func(x, y, sstride, dstride, sdata, ddata);
}
