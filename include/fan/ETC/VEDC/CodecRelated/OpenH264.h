sint32_t
ETC_VEDC_CodecRelated_OpenH264_ToInternal_RateControl(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_EncoderSetting_RateControl_t *RateControl
){
  SEncParamExt *InternalSetting = &Encoder->Codec.OpenH264.InternalSetting;

  switch(RateControl->Type){
    case ETC_VEDC_EncoderSetting_RateControlType_VBR:{
      InternalSetting->iRCMode = RC_BITRATE_MODE;
      InternalSetting->iTargetBitrate = RateControl->VBR.bps;
      return 0;
    }
    case ETC_VEDC_EncoderSetting_RateControlType_TBR:{
      InternalSetting->iRCMode = RC_TIMESTAMP_MODE;
      InternalSetting->iTargetBitrate = RateControl->TBR.bps;
      return 0;
    }
    default:{
      return -1;
    }
  }
}

sint32_t
ETC_VEDC_CodecRelated_OpenH264_ToInternal_EncoderSetting(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_EncoderSetting_t *Setting
){
  SEncParamExt *InternalSetting = &Encoder->Codec.OpenH264.InternalSetting;

  Encoder->Codec.OpenH264.en->GetDefaultParams(InternalSetting);

  {
    /* hardcore value setting */
    /* its forced because openh264's default settings are not acceptable */

    /* default value for libx264 */
    InternalSetting->iMinQp = 23;
  }

  switch(Setting->UsageType){
    case ETC_VEDC_EncoderSetting_UsageType_Realtime:{
      InternalSetting->iUsageType = CAMERA_VIDEO_REAL_TIME;
      break;
    }
    default:{
      return -1;
    }
  }

  InternalSetting->iPicWidth = Setting->FrameSizeX;
  InternalSetting->iPicHeight = Setting->FrameSizeY;

  {
    sint32_t ir = ETC_VEDC_CodecRelated_OpenH264_ToInternal_RateControl(Encoder, &Setting->RateControl);
    if(ir != 0){
      return ir;
    }
  }

  InternalSetting->fMaxFrameRate = Setting->InputFrameRate;

  return 0;
}

sint32_t
ETC_VEDC_CodecRelated_OpenH264_Allocate(
  ETC_VEDC_Encoder_t *Encoder
){
  int ir; /* internal return */

  ir = WelsCreateSVCEncoder(&Encoder->Codec.OpenH264.en);
  if(ir != 0){
    return 1;
  }

  return 0;
}
void
ETC_VEDC_CodecRelated_OpenH264_Deallocate(
  ETC_VEDC_Encoder_t *Encoder
){
  WelsDestroySVCEncoder(Encoder->Codec.OpenH264.en);
}

sint32_t
ETC_VEDC_CodecRelated_OpenH264_Open(
  ETC_VEDC_Encoder_t *Encoder
){
  int ir = Encoder->Codec.OpenH264.en->InitializeExt(&Encoder->Codec.OpenH264.InternalSetting);
  if(ir != 0){
    return 1;
  }

  return 0;
}
void
ETC_VEDC_CodecRelated_OpenH264_Close(
  ETC_VEDC_Encoder_t *Encoder
){
  int ir = Encoder->Codec.OpenH264.en->Uninitialize();
  if(ir != 0){
    PR_abort();
  }
}
