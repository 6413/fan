#pragma once

#include _WITCH_PATH(ETC/PIXF/PIXF.h)

#include <wels/codec_api.h>

typedef enum{
  ETC_VEDC_DecoderID_OpenH264
}ETC_VEDC_DecoderID;

typedef struct{
  ETC_VEDC_DecoderID DecoderID;

  union{
    struct{
      ISVCDecoder *de;
      struct{
        DECODING_STATE state;
        SBufferInfo BufferInfo;
        uint8_t *Data[4];
      }wrd;
    }OpenH264;
  }Codec;
}ETC_VEDC_Decoder_t;

sint32_t
ETC_VEDC_Decoder_Open(
  ETC_VEDC_Decoder_t *Decoder,
  ETC_VEDC_DecoderID DecoderID
){
  Decoder->DecoderID = DecoderID;

  switch(Decoder->DecoderID){
    case ETC_VEDC_DecoderID_OpenH264:{
      int ir; /* internal return */

      ir = WelsCreateDecoder(&Decoder->Codec.OpenH264.de);
      if(ir != 0){
        return 1;
      }

      {
        SDecodingParam InternalSetting = {0};
        InternalSetting.eEcActiveIdc = ERROR_CON_SLICE_MV_COPY_CROSS_IDR;
        long r = Decoder->Codec.OpenH264.de->Initialize(&InternalSetting);
        if(r != 0){
          return -1;
        }
      }

      return 0;
    }
    default:{
      return 1;
    }
  }
}
void
ETC_VEDC_Decoder_Close(
  ETC_VEDC_Decoder_t *Decoder
){
  switch(Decoder->DecoderID){
    case ETC_VEDC_DecoderID_OpenH264:{
      PR_abort();
      return;
    }
  }
}

sint32_t
ETC_VEDC_Decoder_Write(
  ETC_VEDC_Decoder_t *Decoder,
  uint8_t *Data,
  uint32_t Size
){
  switch(Decoder->DecoderID){
    case ETC_VEDC_DecoderID_OpenH264:{
      Decoder->Codec.OpenH264.wrd.state = Decoder->Codec.OpenH264.de->DecodeFrameNoDelay(
        Data,
        Size,
        Decoder->Codec.OpenH264.wrd.Data,
        &Decoder->Codec.OpenH264.wrd.BufferInfo);
      return 0;
    }
  }
}

typedef struct{
  ETC_PIXF PixelFormat;
  void *Data[4];
  uint32_t Stride[4];
  uint32_t SizeX;
  uint32_t SizeY;
}ETC_VEDC_Decoder_Frame_t;

/* returns true if there is frame */
bool
ETC_VEDC_Decoder_Read(
  ETC_VEDC_Decoder_t *Decoder,
  ETC_VEDC_Decoder_Frame_t *Frame
){
  switch(Decoder->DecoderID){
    case ETC_VEDC_DecoderID_OpenH264:{
      if(Decoder->Codec.OpenH264.wrd.BufferInfo.iBufferStatus == 0){
        return 0;
      }

      if(Decoder->Codec.OpenH264.wrd.BufferInfo.UsrData.sSystemBuffer.iFormat != videoFormatI420){
        return 0;
      }

      Frame->PixelFormat = ETC_PIXF_YUV420p;

      Frame->Data[0] = Decoder->Codec.OpenH264.wrd.Data[0];
      Frame->Data[1] = Decoder->Codec.OpenH264.wrd.Data[1];
      Frame->Data[2] = Decoder->Codec.OpenH264.wrd.Data[2];

      Frame->Stride[0] = Decoder->Codec.OpenH264.wrd.BufferInfo.UsrData.sSystemBuffer.iStride[0];
      Frame->Stride[1] = Decoder->Codec.OpenH264.wrd.BufferInfo.UsrData.sSystemBuffer.iStride[1];
      Frame->Stride[2] = Decoder->Codec.OpenH264.wrd.BufferInfo.UsrData.sSystemBuffer.iStride[1];

      Frame->SizeX = Decoder->Codec.OpenH264.wrd.BufferInfo.UsrData.sSystemBuffer.iWidth;
      Frame->SizeY = Decoder->Codec.OpenH264.wrd.BufferInfo.UsrData.sSystemBuffer.iHeight;

      return 1;
    }
  }
}
