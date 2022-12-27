sint32_t
ETC_VEDC_CodecRelated_x264_ToInternal_RateControl(
  ETC_VEDC_Encoder_t *Encoder,
  ETC_VEDC_EncoderSetting_RateControl_t *RateControl
){
  x264_param_t *InternalSetting = &Encoder->Codec.x264.InternalSetting;

  InternalSetting->i_nal_hrd = X264_NAL_HRD_CBR;
  InternalSetting->rc.i_rc_method = X264_RC_ABR;

  switch(RateControl->Type){
    case ETC_VEDC_EncoderSetting_RateControlType_VBR:{
      InternalSetting->rc.i_bitrate = RateControl->VBR.bps / 1024;
      InternalSetting->rc.i_vbv_max_bitrate = RateControl->VBR.bps / 1024;
      InternalSetting->rc.i_vbv_buffer_size = RateControl->VBR.bps / 1024;
      return 0;
    }
    case ETC_VEDC_EncoderSetting_RateControlType_TBR:{
      InternalSetting->rc.i_bitrate = RateControl->TBR.bps / 1024;
      InternalSetting->rc.i_vbv_max_bitrate = RateControl->TBR.bps / 1024;
      InternalSetting->rc.i_vbv_buffer_size = RateControl->TBR.bps / 1024;
      return 0;
    }
    default:{
      return -1;
    }
  }
}
