#pragma once

#include _WITCH_PATH(A/A.h)
#include _WITCH_PATH(ETC/CC/st/st.h)
#include _WITCH_PATH(ETC/PIXF/PIXF.h)

#include "DefineCodecs.h"

typedef enum{
  ETC_VEDC_EncoderID_Nothing,
  ETC_VEDC_EncoderID_OpenH264,
  ETC_VEDC_EncoderID_x264
}ETC_VEDC_EncoderID;

ETC_VEDC_EncoderID ETC_VEDC_Encoder_IDFromString(uintptr_t Size, const uint8_t *String){
  const uint8_t Nothing[] = {'N','o','t','h','i','n','g'};
  if(Size == sizeof(Nothing) && STR_ncasecmp(String, Nothing, Size) == 0) return ETC_VEDC_EncoderID_Nothing;
  const uint8_t OpenH264[] = {'O','p','e','n','H','2','6','4'};
  if(Size == sizeof(OpenH264) && STR_ncasecmp(String, OpenH264, Size) == 0) return ETC_VEDC_EncoderID_OpenH264;
  const uint8_t x264[] = {'x','2','6','4'};
  if(Size == sizeof(x264) && STR_ncasecmp(String, x264, Size) == 0) return ETC_VEDC_EncoderID_x264;

  return ETC_VEDC_EncoderID_Nothing;
}

typedef struct{
  ETC_VEDC_EncoderID EncoderID;

  union{
    struct{
      ISVCEncoder *en;
      SEncParamExt InternalSetting;
      struct{
        SFrameBSInfo info;
        uint32_t i;
      }wrd;
    }OpenH264;
    struct{
      x264_t *en;
      x264_param_t InternalSetting;
      struct{
        x264_nal_t *pnal;
        int inal;
        uint32_t i;
      }wrd;
    }x264;
  }Codec;
}ETC_VEDC_Encoder_t;

typedef enum{
  ETC_VEDC_EncoderSetting_UsageType_Realtime
}ETC_VEDC_EncoderSetting_UsageType;

typedef enum{
  ETC_VEDC_EncoderSetting_RateControlType_VBR, /* variable based bit rate */
  ETC_VEDC_EncoderSetting_RateControlType_TBR /* time based bit rate */
}ETC_VEDC_EncoderSetting_RateControlType;

typedef struct{
  ETC_VEDC_EncoderSetting_RateControlType Type;
  union{
    struct{
      uint32_t bps; /* bit per second */
    }VBR;
    struct{
      uint32_t bps; /* bit per second */
    }TBR;
  };
}ETC_VEDC_EncoderSetting_RateControl_t;

bool
ETC_VEDC_EncoderSetting_RateControl_IsEqual(
  ETC_VEDC_EncoderSetting_RateControl_t *RateControl0,
  ETC_VEDC_EncoderSetting_RateControl_t *RateControl1
){
  ETC_VEDC_EncoderSetting_RateControlType Type = RateControl0->Type;
  if(Type != RateControl1->Type){
    return 0;
  }

  switch(Type){
    case ETC_VEDC_EncoderSetting_RateControlType_VBR:{
      return RateControl0->VBR.bps == RateControl1->VBR.bps;
    }
    case ETC_VEDC_EncoderSetting_RateControlType_TBR:{
      return RateControl0->TBR.bps == RateControl1->TBR.bps;
    }
  }

  return 1;
}

typedef struct{
  ETC_VEDC_EncoderSetting_UsageType UsageType;
  uint32_t FrameSizeX;
  uint32_t FrameSizeY;
  ETC_VEDC_EncoderSetting_RateControl_t RateControl;
  f32_t InputFrameRate;
}ETC_VEDC_EncoderSetting_t;

#include "CodecRelated/CodecRelated.h"

sint32_t
ETC_VEDC_Encoder_Open(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_EncoderSetting_t *Setting,
  ETC_VEDC_EncoderID EncoderID
){
  Encoder->EncoderID = EncoderID;

  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      sint32_t ir;

      ir = ETC_VEDC_CodecRelated_OpenH264_Allocate(Encoder);
      if(ir != 0){
        return ir;
      }

      ir = ETC_VEDC_CodecRelated_OpenH264_ToInternal_EncoderSetting(Encoder, Setting);
      if(ir != 0){
        return ir;
      }

      ir = ETC_VEDC_CodecRelated_OpenH264_Open(Encoder);
      if(ir != 0){
        ETC_VEDC_CodecRelated_OpenH264_Deallocate(Encoder);
        return ir;
      }

      return 0;
    }
    case ETC_VEDC_EncoderID_x264:{
      if(x264_param_default_preset(&Encoder->Codec.x264.InternalSetting, "veryfast", "zerolatency") != 0){
        return 1;
      }
      if(x264_param_apply_profile(&Encoder->Codec.x264.InternalSetting, "high") != 0){
        return 1;
      }

      Encoder->Codec.x264.InternalSetting.i_csp = X264_CSP_I420;
      Encoder->Codec.x264.InternalSetting.i_width = Setting->FrameSizeX;
      Encoder->Codec.x264.InternalSetting.i_height = Setting->FrameSizeY;
      Encoder->Codec.x264.InternalSetting.b_vfr_input = 0;
      Encoder->Codec.x264.InternalSetting.b_repeat_headers = 1;
      Encoder->Codec.x264.InternalSetting.b_annexb = 1;

      Encoder->Codec.x264.InternalSetting.i_fps_num = Setting->InputFrameRate;
      Encoder->Codec.x264.InternalSetting.i_fps_den = 1;
      Encoder->Codec.x264.InternalSetting.i_timebase_num = 1000;
      Encoder->Codec.x264.InternalSetting.i_timebase_den = 1;

      Encoder->Codec.x264.InternalSetting.rc.i_qp_min = 23;
      {
        sint32_t r = ETC_VEDC_CodecRelated_x264_ToInternal_RateControl(Encoder, &Setting->RateControl);
        if(r != 0){
          return r;
        }
      }
      Encoder->Codec.x264.en = x264_encoder_open(&Encoder->Codec.x264.InternalSetting);
      if(Encoder->Codec.x264.en == NULL){
        return 1;
      }
      return 0;
    }
    default:{
      return 1;
    }
  }
}
void
ETC_VEDC_Encoder_Close(
  ETC_VEDC_Encoder_t *Encoder
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      ETC_VEDC_CodecRelated_OpenH264_Close(Encoder);
      ETC_VEDC_CodecRelated_OpenH264_Deallocate(Encoder);
      return;
    }
    case ETC_VEDC_EncoderID_x264:{
      x264_encoder_close(Encoder->Codec.x264.en);
      return;
    }
  }
}

sint32_t
ETC_VEDC_Encoder_Set_FrameSize(
  ETC_VEDC_Encoder_t *Encoder,
  uint32_t FrameSizeX,
  uint32_t FrameSizeY
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      ETC_VEDC_CodecRelated_OpenH264_Close(Encoder);

      Encoder->Codec.OpenH264.InternalSetting.iPicWidth = FrameSizeX;
      Encoder->Codec.OpenH264.InternalSetting.iPicHeight = FrameSizeY;

      sint32_t ir = ETC_VEDC_CodecRelated_OpenH264_Open(Encoder);
      if(ir != 0){
        ETC_VEDC_CodecRelated_OpenH264_Deallocate(Encoder);
        return ir;
      }

      return 0;
    }
    case ETC_VEDC_EncoderID_x264:{
      x264_encoder_close(Encoder->Codec.x264.en);
      Encoder->Codec.x264.InternalSetting.i_width = FrameSizeX;
      Encoder->Codec.x264.InternalSetting.i_height = FrameSizeY;
      Encoder->Codec.x264.en = x264_encoder_open(&Encoder->Codec.x264.InternalSetting);
      if(Encoder->Codec.x264.en == NULL){
        return 1;
      }
      return 0;
    }
  }
}

sint32_t
ETC_VEDC_Encoder_Set_RateControl(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_EncoderSetting_RateControl_t *RateControl
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      ETC_VEDC_CodecRelated_OpenH264_Close(Encoder);

      sint32_t ir;

      ir = ETC_VEDC_CodecRelated_OpenH264_ToInternal_RateControl(Encoder, RateControl);
      if(ir != 0){
        ETC_VEDC_CodecRelated_OpenH264_Deallocate(Encoder);
        return ir;
      }

      ir = ETC_VEDC_CodecRelated_OpenH264_Open(Encoder);
      if(ir != 0){
        ETC_VEDC_CodecRelated_OpenH264_Deallocate(Encoder);
        return ir;
      }

      return 0;
    }
    case ETC_VEDC_EncoderID_x264:{
      x264_encoder_close(Encoder->Codec.x264.en);
      {
        sint32_t r = ETC_VEDC_CodecRelated_x264_ToInternal_RateControl(Encoder, RateControl);
        if(r != 0){
          return r;
        }
      }
      Encoder->Codec.x264.en = x264_encoder_open(&Encoder->Codec.x264.InternalSetting);
      if(Encoder->Codec.x264.en == NULL){
        return 1;
      }
      return 0;
    }
  }
}

sint32_t
ETC_VEDC_Encoder_Set_InputFrameRate(
  ETC_VEDC_Encoder_t *Encoder,
  f32_t InputFrameRate
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      ETC_VEDC_CodecRelated_OpenH264_Close(Encoder);

      Encoder->Codec.OpenH264.InternalSetting.fMaxFrameRate = InputFrameRate;

      sint32_t ir = ETC_VEDC_CodecRelated_OpenH264_Open(Encoder);
      if(ir != 0){
        ETC_VEDC_CodecRelated_OpenH264_Deallocate(Encoder);
        return ir;
      }

      return 0;
    }
    case ETC_VEDC_EncoderID_x264:{
      x264_encoder_close(Encoder->Codec.x264.en);
      Encoder->Codec.x264.InternalSetting.i_fps_num = InputFrameRate;
      Encoder->Codec.x264.en = x264_encoder_open(&Encoder->Codec.x264.InternalSetting);
      if(Encoder->Codec.x264.en == NULL){
        return 1;
      }
      return 0;
    }
  }
}

typedef struct{
  ETC_PIXF PixelFormat;
  uint32_t Stride[4];
  void *Data[4];
  uint32_t SizeX;
  uint32_t SizeY;
  uint64_t TimeStamp;

  struct{
    bool Alloced;
  }_Optimize;
}ETC_VEDC_Encoder_Source_t;

/* converts source to best pixel format for encoder */
void
ETC_VEDC_Encoder_Source_Optimize(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_Encoder_Source_t *Source
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_OpenH264:
    case ETC_VEDC_EncoderID_x264:
    {
      uint32_t Stride[4] = {Source->SizeX, Source->SizeX / 2, Source->SizeX / 2, 0};
      void *Data[4];
      Data[0] = A_resize(0, Stride[0] * Source->SizeY);
      Data[1] = A_resize(0, Stride[1] * (Source->SizeY / 2));
      Data[2] = A_resize(0, Stride[2] * (Source->SizeY / 2));

      CC_st_convert(
        Source->PixelFormat,
        ETC_PIXF_YUV420p,
        Source->SizeX,
        Source->SizeY,
        (const uint32_t *)Source->Stride,
        Stride,
        (const uint8_t *const *)Source->Data,
        (uint8_t *const *)&Data);

      for(uint8_t i = 0; i < 4; i++) { Source->Data[i] = Data[i]; }
      for(uint8_t i = 0; i < 4; i++) { Source->Stride[i] = Stride[i]; }

      Source->PixelFormat = ETC_PIXF_YUV420p;

      Source->_Optimize.Alloced = 1;

      return;
    }
  }
}

/* must be called if Encoder_Source_Optimize called previously */
void
ETC_VEDC_Encoder_Source_Close(
  ETC_VEDC_Encoder_Source_t *Source
){
  if(Source->_Optimize.Alloced == 0){
    return;
  }

  switch(Source->PixelFormat){
    case ETC_PIXF_YUV420p:{
      A_resize(Source->Data[0], 0);
      A_resize(Source->Data[1], 0);
      A_resize(Source->Data[2], 0);
      break;
    }
  }
}

sint32_t
ETC_VEDC_Encoder_Write(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_Encoder_Source_t *Source
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      memset(&Encoder->Codec.OpenH264.wrd.info, 0, sizeof(SFrameBSInfo));
      Encoder->Codec.OpenH264.wrd.i = 0;

      SSourcePicture InternalSource;

      switch(Source->PixelFormat){
        case ETC_PIXF_YUV420p:{
          InternalSource.iColorFormat = videoFormatI420;
          break;
        }
        default:{
          return -1;
        }
      }

      for(uint8_t i = 0; i < 4; i++) { InternalSource.iStride[i] = Source->Stride[i]; }
      for(uint8_t i = 0; i < 4; i++) { InternalSource.pData[i] = (unsigned char *)Source->Data[i]; }

      InternalSource.iPicWidth = Source->SizeX;
      InternalSource.iPicHeight = Source->SizeY;

      InternalSource.uiTimeStamp = Source->TimeStamp / 1000000;

      int ir = Encoder->Codec.OpenH264.en->EncodeFrame(&InternalSource, &Encoder->Codec.OpenH264.wrd.info);
      if(ir != 0){
        return -1;
      }

      return 0;
    }
    case ETC_VEDC_EncoderID_x264:{
      Encoder->Codec.x264.wrd.i = 0;

      x264_picture_t pic;
      x264_picture_init(&pic);

      switch(Source->PixelFormat){
        case ETC_PIXF_YUV420p:{
          pic.img.i_csp = X264_CSP_I420;
          break;
        }
        default:{
          return -1;
        }
      }

      for(uint8_t i = 0; i < 4; i++) { pic.img.i_stride[i] = Source->Stride[i]; }
      for(uint8_t i = 0; i < 4; i++) { pic.img.plane[i] = (uint8_t *)Source->Data[i]; }

      pic.i_pts = Source->TimeStamp / 1000000;

      x264_picture_t pic_filler;
      int r = x264_encoder_encode(
        Encoder->Codec.x264.en,
        &Encoder->Codec.x264.wrd.pnal,
        &Encoder->Codec.x264.wrd.inal,
        &pic,
        &pic_filler);
      if(r < 0){
        return 1;
      }

      return 0;
    }
  }
}

bool
ETC_VEDC_Encoder_IsReadAble(
  ETC_VEDC_Encoder_t *Encoder
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      return Encoder->Codec.OpenH264.wrd.info.iLayerNum != 0;
    }
    case ETC_VEDC_EncoderID_x264:{
      return Encoder->Codec.x264.wrd.inal != 0;
    }
  }
}

/* this function should be called in while(f(...)) till return is 0 */
/*
  its undefined to call this function:
    - before Write
    - after it returned 0
*/
uint32_t /* output size */
ETC_VEDC_Encoder_Read(
  ETC_VEDC_Encoder_t *Encoder,
  void **Data /* will point where is output */
){
  switch(Encoder->EncoderID){
    case ETC_VEDC_EncoderID_Nothing:{
      return 0;
    }
    case ETC_VEDC_EncoderID_OpenH264:{
      if(Encoder->Codec.OpenH264.wrd.i == Encoder->Codec.OpenH264.wrd.info.iLayerNum){
        return 0;
      }

      SLayerBSInfo *linfo = &Encoder->Codec.OpenH264.wrd.info.sLayerInfo[Encoder->Codec.OpenH264.wrd.i];

      uint32_t r = 0;
      for(uint32_t iNal = 0; iNal < linfo->iNalCount; iNal++){
        r += linfo->pNalLengthInByte[iNal];
      }

      *Data = linfo->pBsBuf;

      Encoder->Codec.OpenH264.wrd.i++;

      return r;
    }
    case ETC_VEDC_EncoderID_x264:{
      if(Encoder->Codec.x264.wrd.i == Encoder->Codec.x264.wrd.inal){
        return 0;
      }

      x264_nal_t *nal = &Encoder->Codec.x264.wrd.pnal[Encoder->Codec.x264.wrd.i++];

      *Data = nal->p_payload;

      return nal->i_payload - nal->i_padding;
    }
  }
}
