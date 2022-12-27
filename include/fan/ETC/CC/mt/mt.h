#pragma once

#include _WITCH_PATH(WITCH.h)
#include _WITCH_PATH(EV/EV.h)

#include _WITCH_PATH(ETC/PIXF/PIXF.h)

/* only useful if you call CC_mt_convert multiple times before relaxing listener */
/* but can give cpu effiency overall */
#ifndef CC_mt_set_active_tp_delete
  #define CC_mt_set_active_tp_delete 1
#endif

typedef struct{
  EV_tp_t tp;
  uint32_t width, height;
  const uint32_t *sstride, *dstride;
  uint8_t *sdata[8], *ddata[8];
  TH_cond_t *cond;
  bool ping;
}_CC_mt_tp_t;
void _CC_mt_thread_end(EV_t *listener, _CC_mt_tp_t *arg){
  TH_lock(&arg->cond->mutex);
  arg->ping = 1;
  TH_signal(arg->cond);
  TH_unlock(&arg->cond->mutex);
}
typedef struct{
  _CC_mt_tp_t *tp;
  TH_cond_t cond;
  uint32_t threads;
}CC_mt_t;
void CC_mt_open(CC_mt_t *ccmt, uint32_t threads){
  ccmt->tp = (_CC_mt_tp_t *)A_resize(0, threads * sizeof(_CC_mt_tp_t));
  TH_cond_init(&ccmt->cond);
  ccmt->threads = threads;
}
void CC_mt_close(CC_mt_t *ccmt){
  A_resize(ccmt->tp, 0);
}

void _CC_mt_wait(CC_mt_t *ccmt, EV_t *listener){
  uint32_t ti = 0;
  TH_lock(&ccmt->cond.mutex);
  while(1){
    while(1){
      if(ccmt->tp[ti].ping){
        #if CC_mt_set_active_tp_delete == 1
          EV_tp_del(listener, ccmt->tp[ti].tp.pool_node);
        #endif
        ti++;
        if(ti == ccmt->threads){
          goto end_gt;
        }
      }
      else{
        break;
      }
    }
    TH_wait(&ccmt->cond);
  }
  end_gt:;
  TH_unlock(&ccmt->cond.mutex);
}

void _CC_mt_tpInside(EV_t *, EV_tp_t *, sint32_t){}

void _CC_mt_open_2_1_1x2(
  CC_mt_t *ccmt,
  EV_t *listener,
  uint32_t width,
  uint32_t height,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata,
  EV_tp_cb_outside_t cb_outside
){
  uint32_t y2 = height / 2;
  uint32_t divy = y2 / ccmt->threads;
  uint32_t left = y2 - divy * ccmt->threads;
  divy = divy * 2 + 2;

  const uint8_t *sdata0 = sdata[0];
  uint8_t *ddata0 = ddata[0];
  uint8_t *ddata1 = ddata[1];

  uint32_t i = 0;
  start_gt:
  for(; i != left; i++){
    ccmt->tp[i].width = width;
    ccmt->tp[i].height = divy;
    ccmt->tp[i].sstride = sstride;
    ccmt->tp[i].dstride = dstride;
    ccmt->tp[i].sdata[0] = (uint8_t *)sdata0;
    ccmt->tp[i].ddata[0] = ddata0;
    ccmt->tp[i].ddata[1] = ddata1;
    ccmt->tp[i].cond = &ccmt->cond;
    ccmt->tp[i].ping = 0;

    EV_tp_init(&ccmt->tp[i].tp, cb_outside, _CC_mt_tpInside, 1);
    EV_tp_start(listener, &ccmt->tp[i].tp);

    sdata0 += divy * sstride[0];
    ddata0 += divy * dstride[0];
    ddata1 += (divy / 2) * dstride[1];
  }
  if(left != ccmt->threads){
    divy -= 2;
    left = ccmt->threads;
    goto start_gt;
  }

  _CC_mt_wait(ccmt, listener);
}

void _CC_mt_open_2_1_1x2x2(
  CC_mt_t *ccmt,
  EV_t *listener,
  uint32_t width,
  uint32_t height,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata,
  EV_tp_cb_outside_t cb_outside
){
  uint32_t y2 = height / 2;
  uint32_t divy = y2 / ccmt->threads;
  uint32_t left = y2 - divy * ccmt->threads;
  divy = divy * 2 + 2;

  const uint8_t *sdata0 = sdata[0];
  uint8_t *ddata0 = ddata[0];
  uint8_t *ddata1 = ddata[1];
  uint8_t *ddata2 = ddata[2];

  uint32_t i = 0;
  start_gt:
  for(; i != left; i++){
    ccmt->tp[i].width = width;
    ccmt->tp[i].height = divy;
    ccmt->tp[i].sstride = sstride;
    ccmt->tp[i].dstride = dstride;
    ccmt->tp[i].sdata[0] = (uint8_t *)sdata0;
    ccmt->tp[i].ddata[0] = ddata0;
    ccmt->tp[i].ddata[1] = ddata1;
    ccmt->tp[i].ddata[2] = ddata2;
    ccmt->tp[i].cond = &ccmt->cond;
    ccmt->tp[i].ping = 0;

    EV_tp_init(&ccmt->tp[i].tp, cb_outside, _CC_mt_tpInside, 1);
    EV_tp_start(listener, &ccmt->tp[i].tp);

    sdata0 += divy * sstride[0];
    ddata0 += divy * dstride[0];
    ddata1 += (divy / 2) * dstride[1];
    ddata2 += (divy / 2) * dstride[2];
  }
  if(left != ccmt->threads){
    divy -= 2;
    left = ccmt->threads;
    goto start_gt;
  }

  _CC_mt_wait(ccmt, listener);
}

void _CC_mt_open_2_1x2x2_1(
  CC_mt_t *ccmt,
  EV_t *listener,
  uint32_t width,
  uint32_t height,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata,
  EV_tp_cb_outside_t cb_outside
){
  uint32_t y2 = height / 2;
  uint32_t divy = y2 / ccmt->threads;
  uint32_t left = y2 - divy * ccmt->threads;
  divy = divy * 2 + 2;

  const uint8_t *sdata0 = sdata[0];
  const uint8_t *sdata1 = sdata[1];
  const uint8_t *sdata2 = sdata[2];
  uint8_t *ddata0 = ddata[0];

  uint32_t i = 0;
  start_gt:
  for(; i != left; i++){
    ccmt->tp[i].width = width;
    ccmt->tp[i].height = divy;
    ccmt->tp[i].sstride = sstride;
    ccmt->tp[i].dstride = dstride;
    ccmt->tp[i].sdata[0] = (uint8_t *)sdata0;
    ccmt->tp[i].sdata[1] = (uint8_t *)sdata1;
    ccmt->tp[i].sdata[2] = (uint8_t *)sdata2;
    ccmt->tp[i].ddata[0] = ddata0;
    ccmt->tp[i].cond = &ccmt->cond;
    ccmt->tp[i].ping = 0;

    EV_tp_init(&ccmt->tp[i].tp, cb_outside, _CC_mt_tpInside, 1);
    EV_tp_start(listener, &ccmt->tp[i].tp);

    sdata0 += divy * sstride[0];
    sdata1 += (divy / 2) * sstride[1];
    sdata2 += (divy / 2) * sstride[2];
    ddata0 += divy * dstride[0];
  }
  if(left != ccmt->threads){
    divy -= 2;
    left = ccmt->threads;
    goto start_gt;
  }

  _CC_mt_wait(ccmt, listener);
}

typedef bool (*_CC_mt_convert_func_t)(
  CC_mt_t *ccmt,
  EV_t *listener,
  uint32_t x,
  uint32_t y,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata
);

#include _WITCH_PATH(ETC/CC/mt/internal/RGB24.h)
#include _WITCH_PATH(ETC/CC/mt/internal/BGRA.h)
#include _WITCH_PATH(ETC/CC/mt/internal/YUV420P.h)

_CC_mt_convert_func_t _CC_mt_getfmt(uint32_t sfmt, uint32_t dfmt){
  switch(sfmt){
    case ETC_PIXF_RGB24:{
      return _CC_mt_RGB24_getfmt(dfmt);
    }
    case ETC_PIXF_BGRA:{
      return _CC_mt_BGRA_getfmt(dfmt);
    }
    case ETC_PIXF_YUV420p:{
      return _CC_mt_YUV420P_getfmt(dfmt);
    }
    default:{
      return 0;
    }
  }
}

bool CC_mt_convert(
  CC_mt_t *ccmt,
  EV_t *listener,
  uint32_t sfmt,
  uint32_t dfmt,
  uint32_t x,
  uint32_t y,
  const uint32_t *sstride,
  const uint32_t *dstride,
  const uint8_t *const *sdata,
  uint8_t *const *ddata
){
  _CC_mt_convert_func_t convert_func = _CC_mt_getfmt(sfmt, dfmt);
  if(!convert_func){
    return 1;
  }
  return convert_func(ccmt, listener, x, y, sstride, dstride, sdata, ddata);
}
