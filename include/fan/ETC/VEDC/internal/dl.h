#pragma once

typedef struct{
  void *f;
  const char *cs;
}_ETC_BVEDC_dl_fst;

#define _ETC_BVEDC_dl_fdef(r, name, ...) \
  struct{ \
    r (*f)(__VA_ARGS__); \
    const char *cs = #name; \
  }name;

struct{
  uint32_t UsedCount;

  void *Handle;

  struct{
    _ETC_BVEDC_dl_fdef(int, Initialize, ISVCEncoder*, const SEncParamBase* pParam);
    _ETC_BVEDC_dl_fdef(int, InitializeExt, ISVCEncoder*, const SEncParamExt* pParam);
    _ETC_BVEDC_dl_fdef(int, GetDefaultParams, ISVCEncoder*, SEncParamExt* pParam);
    _ETC_BVEDC_dl_fdef(int, Uninitialize, ISVCEncoder*);
    _ETC_BVEDC_dl_fdef(int, EncodeFrame, ISVCEncoder*, const SSourcePicture* kpSrcPic, SFrameBSInfo* pBsInfo);
    _ETC_BVEDC_dl_fdef(int, EncodeParameterSets, ISVCEncoder*, SFrameBSInfo* pBsInfo);
    _ETC_BVEDC_dl_fdef(int, ForceIntraFrame, ISVCEncoder*, bool bIDR);
    _ETC_BVEDC_dl_fdef(int, SetOption, ISVCEncoder*, ENCODER_OPTION eOptionId, void* pOption);
    _ETC_BVEDC_dl_fdef(int, GetOption, ISVCEncoder*, ENCODER_OPTION eOptionId, void* pOption);
  }Encoder;
}_ETC_BVEDC_dl = {
  .UsedCount = 0
};

#undef _ETC_BVEDC_dl_fdef

void _ETC_BVEDC_dl_inc(){
  if(_ETC_BVEDC_dl.UsedCount == 0){
    _ETC_BVEDC_dl.Handle = dlopen("libopenh264.so", RTLD_LAZY);
    if(_ETC_BVEDC_dl.Handle == NULL){
      return;
    }

    uint32_t to;

    to = sizeof(_ETC_BVEDC_dl.Encoder) / sizeof(_ETC_BVEDC_dl_fst);
    for(uint32_t i = 0; i < to; i++){
      _ETC_BVEDC_dl_fst *fst = &((_ETC_BVEDC_dl_fst *)&_ETC_BVEDC_dl.Encoder)[i];
      fst.f = dlsym(handle, fst->cs);
      if(fst.f == NULL){
        /* not found */
      }
    }
  }

  _ETC_BVEDC_dl.UsedCount++;
}
void _ETC_BVEDC_dl_dec(){
  _ETC_BVEDC_dl.UsedCount--;
}
