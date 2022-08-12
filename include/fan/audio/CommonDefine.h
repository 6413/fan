void _audio_common_close(_audio_common_t *audio_common) {
  _PlayInfoList_close(&audio_common->PlayInfoList);
  A_resize(audio_common->GroupList, 0);
  VEC_free(&audio_common->PlayList);
  VEC_free(&audio_common->MessageQueueList);
}
void _audio_common_open(_audio_common_t *audio_common, uint32_t GroupAmount) {
  TH_mutex_init(&audio_common->PlayInfoListMutex);
  _PlayInfoList_open(&audio_common->PlayInfoList);

  audio_common->GroupAmount = GroupAmount;
  audio_common->GroupList = (_Group_t *)A_resize(0, sizeof(_Group_t) * audio_common->GroupAmount);
  for (uint32_t i = 0; i < audio_common->GroupAmount; i++) {
    audio_common->GroupList[i].FirstReference = _PlayInfoList_NewNodeLast_alloc(&audio_common->PlayInfoList);
    audio_common->GroupList[i].LastReference = _PlayInfoList_NewNodeLast_alloc(&audio_common->PlayInfoList);
  }

  VEC_init(&audio_common->PlayList, sizeof(_Play_t), A_resize);

  TH_mutex_init(&audio_common->MessageQueueListMutex);
  VEC_init(&audio_common->MessageQueueList, sizeof(_Message_t), A_resize);

  for(uint32_t Channel = 0; Channel < _constants::Opus::SupportedChannels; Channel++){
    uint32_t size = sizeof(_DecoderHead_t);
    size += opus_decoder_get_size(Channel + 1);
    _DecoderList_t *dl = &audio_common->DecoderList[Channel];
    _DecoderList_open(dl, size);
    for(uint32_t i = 0; i < _constants::Opus::CacheDecoderPerChannel; i++){
      _DecoderList_NodeReference_t r = _DecoderList_NewNodeLast_alloc(dl);
      uint8_t *dld = (uint8_t *)_DecoderList_GetNodeReferenceData(dl, r);

      _DecoderHead_t *DecoderHead = (_DecoderHead_t *)dld;
      DecoderHead->CacheID = _CacheList_GetConstantInvalidNodeReference();

      OpusDecoder *od = (OpusDecoder *)&dld[sizeof(_DecoderHead_t)];
      int re = opus_decoder_init(od, 48000, Channel + 1);
      if(re != OPUS_OK){
        fan::throw_error("opus_decoder_init");
      }
    }
  }

  _CacheList_open(&audio_common->CacheList);
  for(uint32_t i = 0; i < _constants::Opus::CacheSegmentAmount; i++){
    _CacheList_NodeReference_t r = _CacheList_NewNodeLast_alloc(&audio_common->CacheList);
    _CacheList_Node_t *n = _CacheList_GetNodeByReference(&audio_common->CacheList, r);
    n->data.SegmentID = (_SegmentID_t)-1;
    n->data.DecoderID = _DecoderList_GetConstantInvalidNodeReference();
  }
}

void _DecodeCopy(
  uint8_t ChannelAmount,
  f32_t *F2_20,
  f32_t *Output
){
  switch(ChannelAmount){
    case 1:{
      /* need manuel interleave */
      for(uint32_t i = 0; i < _constants::Opus::SegmentFrameAmount20; i++){
        Output[i * 2 + 0] = F2_20[i];
        Output[i * 2 + 1] = F2_20[i];
      }
      break;
    }
    case 2:{
      MEM_copy(
        F2_20,
        Output,
        _constants::Opus::SegmentFrameAmount20 * 2 * sizeof(f32_t));
      break;
    }
  }
}

_CacheID_t _GetCacheID(_audio_common_t *audio_common, piece_t *piece, _SegmentID_t SegmentID){
  _CacheID_t clid = _CacheList_GetNodeLast(&audio_common->CacheList);
  _CacheList_Node_t *cln = _CacheList_GetNodeByReference(&audio_common->CacheList, clid);
  if(cln->data.SegmentID != (_SegmentID_t)-1){
    ((_SACSegment_t *)cln->data.piece->SACData)[cln->data.SegmentID].CacheID =
      _CacheList_GetConstantInvalidNodeReference();
    _DecoderID_t dlid = cln->data.DecoderID;
    if(_DecoderList_IsNodeReferenceInvalidConstant(dlid) == false){
      _DecoderList_t *dl = &audio_common->DecoderList[cln->data.piece->ChannelAmount - 1];
      _DecoderHead_t *DecoderHead = (_DecoderHead_t *)_DecoderList_GetNodeReferenceData(
        dl,
        dlid);
      DecoderHead->CacheID = _CacheList_GetConstantInvalidNodeReference();
      _DecoderList_ReLinkAsLast(dl, dlid);
    }
  }
  cln->data.piece = piece;
  cln->data.SegmentID = SegmentID;
  ((_SACSegment_t *)piece->SACData)[SegmentID].CacheID = clid;
  return clid;
}

void _WarmUpDecoder(_audio_common_t *audio_common, piece_t *piece, _SegmentID_t SegmentID, OpusDecoder *od){
  opus_decoder_ctl(od, OPUS_RESET_STATE);

  _SegmentID_t fsid = SegmentID - _constants::Opus::DecoderWarmUpAmount;
  if(SegmentID < _constants::Opus::DecoderWarmUpAmount){
    fsid = 0;
  }

  while(fsid != SegmentID){
    _SACSegment_t *SACSegment = &((_SACSegment_t *)piece->SACData)[fsid];

    f32_t F2_20[_constants::Opus::SegmentFrameAmount20 * 5 * 2];
    sint32_t oerr = opus_decode_float(
      od,
      &piece->SACData[SACSegment->Offset],
      SACSegment->Size,
      F2_20,
      _constants::Opus::SegmentFrameAmount20 * 5,
      0);

    if(oerr != _constants::Opus::SegmentFrameAmount20){
      fan::print("help", oerr);
      fan::throw_error("a");
    }

    fsid++;
  }
}

_DecoderID_t _GetDecoderID(_audio_common_t *audio_common, piece_t *piece, _SegmentID_t SegmentID){
  _DecoderList_t *dl = &audio_common->DecoderList[piece->ChannelAmount - 1];

  if(SegmentID != 0){
    _CacheID_t pclid = ((_SACSegment_t *)piece->SACData)[SegmentID - 1].CacheID;
    if(_CacheList_IsNodeReferenceInvalidConstant(pclid) == false){
      _CacheList_Node_t *pcln = _CacheList_GetNodeByReference(&audio_common->CacheList, pclid);
      _DecoderID_t DecoderID = pcln->data.DecoderID;
      if(_DecoderList_IsNodeReferenceInvalidConstant(DecoderID) == false){
        pcln->data.DecoderID = _DecoderList_GetConstantInvalidNodeReference();
        return DecoderID;
      }
    }
  }

  _DecoderID_t dlid = _DecoderList_GetNodeLast(dl);

  uint8_t *dld = (uint8_t *)_DecoderList_GetNodeReferenceData(dl, dlid);

  _DecoderHead_t *DecoderHead = (_DecoderHead_t *)dld;

  _CacheID_t dclid = DecoderHead->CacheID;
  if(_CacheList_IsNodeReferenceInvalidConstant(dclid) == false){
    _CacheList_Node_t *dcln = _CacheList_GetNodeByReference(&audio_common->CacheList, dclid);
    dcln->data.DecoderID = _DecoderList_GetConstantInvalidNodeReference();
  }

  OpusDecoder *od = (OpusDecoder *)&dld[sizeof(_DecoderHead_t)];

  _WarmUpDecoder(audio_common, piece, SegmentID, od);

  return dlid;
}

void _LinkDecoderCache(_audio_common_t *audio_common, _DecoderList_t *dl, _DecoderID_t DecoderID, _CacheID_t CacheID){
  _DecoderHead_t *DecoderHead = (_DecoderHead_t *)_DecoderList_GetNodeReferenceData(dl, DecoderID);
  DecoderHead->CacheID = CacheID;
  _CacheList_Node_t *cln = _CacheList_GetNodeByReference(&audio_common->CacheList, CacheID);
  cln->data.DecoderID = DecoderID;
}

void _DecodeSegment(_audio_common_t *audio_common, piece_t *piece, _SegmentID_t SegmentID){
  _DecoderList_t *dl = &audio_common->DecoderList[piece->ChannelAmount - 1];

  _DecoderID_t DecoderID = _GetDecoderID(audio_common, piece, SegmentID);
  _DecoderList_ReLinkAsFirst(dl, DecoderID);

  _CacheID_t CacheID = _GetCacheID(audio_common, piece, SegmentID);

  _LinkDecoderCache(audio_common, dl, DecoderID, CacheID);

  OpusDecoder *od = (OpusDecoder *)&((uint8_t *)_DecoderList_GetNodeReferenceData(
    dl,
    DecoderID))[sizeof(_DecoderHead_t)];

  _SACSegment_t *SACSegment = &((_SACSegment_t *)piece->SACData)[SegmentID];

  f32_t F2_20[_constants::Opus::SegmentFrameAmount20 * 5 * 2];
  sint32_t oerr = opus_decode_float(
    od,
    &piece->SACData[SACSegment->Offset],
    SACSegment->Size,
    F2_20,
    _constants::Opus::SegmentFrameAmount20 * 5,
    0);

  if(oerr != _constants::Opus::SegmentFrameAmount20){
    fan::print("help", oerr);
    fan::throw_error("a");
  }

  _CacheList_Node_t *cln = _CacheList_GetNodeByReference(&audio_common->CacheList, CacheID);
  _DecodeCopy(piece->ChannelAmount, F2_20, cln->data.Samples);
}

void _GetFrames(_audio_common_t *audio_common, piece_t *piece, uint64_t Offset, f32_t **FramePointer, uint32_t *FrameAmount) {
  Offset += piece->BeginCut;
  _SegmentID_t SegmentID = Offset / _constants::FrameCacheAmount;
  uint32_t PieceCacheMod = Offset % _constants::FrameCacheAmount;
  _CacheID_t *CacheID = &((_SACSegment_t *)piece->SACData)[SegmentID].CacheID;
  if(_CacheList_IsNodeReferenceInvalidConstant(*CacheID) == true){
    _DecodeSegment(audio_common, piece, SegmentID);
  }
  _CacheList_ReLinkAsFirst(&audio_common->CacheList, *CacheID);
  _CacheList_Node_t *cln = _CacheList_GetNodeByReference(&audio_common->CacheList, *CacheID);
  *FramePointer = &cln->data.Samples[PieceCacheMod * _constants::ChannelAmount];
  *FrameAmount = _constants::FrameCacheAmount - PieceCacheMod;
}

void _AddSoundToPlay(_audio_common_t *audio_common, _PlayInfoList_NodeReference_t PlayInfoReference) {
  VEC_handle0(&audio_common->PlayList, 1);
  uint32_t PlayID = audio_common->PlayList.Current - 1;
  _Play_t *Play = &((_Play_t *)audio_common->PlayList.ptr)[PlayID];
  Play->Reference = PlayInfoReference;
  _PlayInfoList_Node_t *PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
  #if fan_debug >= 0
    if (PlayInfoNode->data.PlayID != (uint32_t)-1) {
      /* trying play sound that already playing */
      fan::throw_error("fan_debug");
    }
  #endif
  PlayInfoNode->data.PlayID = PlayID;
}
void
_RemoveFromPlayList
(
  _audio_common_t *audio_common,
  uint32_t PlayID
){
  /* super fast remove */
  ((_Play_t *)audio_common->PlayList.ptr)[PlayID] = ((_Play_t *)audio_common->PlayList.ptr)[--audio_common->PlayList.Current];

  /* moved one needs update */
  _PlayInfoList_NodeReference_t PlayInfoReference = ((_Play_t *)audio_common->PlayList.ptr)[PlayID].Reference;
  _PlayInfoList_Node_t *PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
  PlayInfoNode->data.PlayID = PlayID;
}
void
_RemoveFromPlayInfoList
(
  _audio_common_t *audio_common,
  _PlayInfoList_NodeReference_t PlayInfoReference,
  const PropertiesSoundStop_t* Properties
){
  _PlayInfoList_Node_t *PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
  if (PlayInfoNode->data.PlayID == (uint32_t)-1) {
    /* properties are ignored */
    TH_lock(&audio_common->PlayInfoListMutex);
    _PlayInfoList_Unlink(&audio_common->PlayInfoList, PlayInfoReference);
    _PlayInfoList_Recycle(&audio_common->PlayInfoList, PlayInfoReference);
    TH_unlock(&audio_common->PlayInfoListMutex);
  }
  else {
    _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
    if (Properties->FadeOutTo != 0) {
      PropertiesSoundPlay_t* PropertiesPlay = &PlayInfoNode->data.properties;
      if (PropertiesPlay->Flags.FadeIn) {
        PropertiesPlay->Flags.FadeIn = false;
        PropertiesPlay->Flags.FadeOut = true;
        f32_t CurrentVolume = PropertiesPlay->FadeFrom / PropertiesPlay->FadeTo;
        PropertiesPlay->FadeTo = Properties->FadeOutTo;
        PropertiesPlay->FadeFrom = ((f32_t)1 - CurrentVolume) * PropertiesPlay->FadeTo;
      }
      else {
        PropertiesPlay->FadeFrom = 0;
        PropertiesPlay->FadeTo = Properties->FadeOutTo;
        PropertiesPlay->Flags.FadeOut = true;
      }
    }
    else {
      _RemoveFromPlayList(audio_common, PlayInfoNode->data.PlayID);
      TH_lock(&audio_common->PlayInfoListMutex);
      _PlayInfoList_Unlink(&audio_common->PlayInfoList, PlayInfoReference);
      _PlayInfoList_Recycle(&audio_common->PlayInfoList, PlayInfoReference);
      TH_unlock(&audio_common->PlayInfoListMutex);
    }
  }
}

sint32_t piece_open(audio_t *audio, fan::audio::piece_t *piece, void *data, uintptr_t size) {
  piece->audio_common = (_audio_common_t *)audio;

  #define tbs(p0) \
    if(DataIndex + (p0) > size){ \
      return -1; \
    }

  uint8_t *Data = (uint8_t *)data;
  uintptr_t DataIndex = 0;

  _SACHead_t *SACHead;
  tbs(sizeof(_SACHead_t));
  SACHead = (_SACHead_t *)&Data[DataIndex];
  DataIndex += sizeof(_SACHead_t);

  if(SACHead->Sign != 0xff){
    return -1;
  }

  piece->ChannelAmount = SACHead->ChannelAmount;

  piece->BeginCut = SACHead->BeginCut;

  tbs(SACHead->TotalSegments);
  uint8_t *SACSegmentSizes = &Data[DataIndex];

  piece->TotalSegments = 0;
  uint64_t TotalOfSACSegmentSizes = 0;
  for(uint32_t i = 0; i < SACHead->TotalSegments; i++){
    TotalOfSACSegmentSizes += SACSegmentSizes[i];
    if(SACSegmentSizes[i] == 0xff){
      continue;
    }
    piece->TotalSegments++;
  }

  uint64_t PieceSegmentSizes = piece->TotalSegments * sizeof(_SACSegment_t);
  {
    DataIndex += SACHead->TotalSegments;
    uint64_t LeftSize = size - DataIndex;
    if(TotalOfSACSegmentSizes != LeftSize){
      fan::throw_error("corrupted sac?");
    }
    uintptr_t ssize = PieceSegmentSizes + LeftSize;
    piece->SACData = A_resize(0, ssize);
    MEM_copy(&Data[DataIndex], &piece->SACData[PieceSegmentSizes], LeftSize);
  }

  {
    uint16_t BeforeSum = 0;
    uint32_t psi = 0;
    uint32_t DataOffset = PieceSegmentSizes;
    for(uint32_t i = 0; i < SACHead->TotalSegments; i++){
      BeforeSum += SACSegmentSizes[i];
      if(SACSegmentSizes[i] == 0xff){
        continue;
      }
      ((_SACSegment_t *)piece->SACData)[psi].Offset = DataOffset;
      ((_SACSegment_t *)piece->SACData)[psi].Size = BeforeSum;
      ((_SACSegment_t *)piece->SACData)[psi].CacheID = _CacheList_GetConstantInvalidNodeReference();
      DataOffset += BeforeSum;
      BeforeSum = 0;
      psi++;
    }
  }

  #undef tbs

  piece->FrameAmount = (uint64_t)piece->TotalSegments * _constants::Opus::SegmentFrameAmount20;
  if(SACHead->EndCut >= piece->FrameAmount){
    return -1;
  }
  piece->FrameAmount -= SACHead->EndCut;

  return 0;
}
sint32_t piece_open(audio_t *audio, fan::audio::piece_t *piece, const std::string &path) {
  sint32_t err;

  std::string sd;
  err = fan::io::file::read(path, &sd);
  if(err != 0){
    return err;
  }

  err = piece_open(audio, piece, sd.data(), sd.size());
  if(err != 0){
    return err;
  }

  return 0;
}

void audio_pause_group(audio_t *audio, uint32_t GroupID) {
  _audio_common_t *audio_common = (_audio_common_t *)audio;
  TH_lock(&audio_common->MessageQueueListMutex);
  VEC_handle0(&audio_common->MessageQueueList, 1);
  _Message_t *Message = &((_Message_t *)audio_common->MessageQueueList.ptr)[audio_common->MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::PauseGroup;
  Message->Data.PauseGroup.GroupID = GroupID;
  TH_unlock(&audio_common->MessageQueueListMutex);
}
void audio_resume_group(audio_t *audio, uint32_t GroupID) {
  _audio_common_t *audio_common = (_audio_common_t *)audio;
  TH_lock(&audio_common->MessageQueueListMutex);
  VEC_handle0(&audio_common->MessageQueueList, 1);
  _Message_t *Message = &((_Message_t*)audio_common->MessageQueueList.ptr)[audio_common->MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::ResumeGroup;
  Message->Data.ResumeGroup.GroupID = GroupID;
  TH_unlock(&audio_common->MessageQueueListMutex);
}

SoundPlayID_t SoundPlay(audio_t *audio, piece_t *piece, uint32_t GroupID, const PropertiesSoundPlay_t *Properties) {
  _audio_common_t *audio_common = (_audio_common_t *)audio;
  #if fan_debug >= 0
    if (GroupID >= audio_common->GroupAmount) {
      fan::throw_error("fan_debug");
    }
  #endif
  TH_lock(&audio_common->PlayInfoListMutex);
  _PlayInfoList_NodeReference_t PlayInfoReference = _PlayInfoList_NewNode(&audio_common->PlayInfoList);
  _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
  PlayInfoNode->data.piece = piece;
  PlayInfoNode->data.GroupID = GroupID;
  PlayInfoNode->data.PlayID = (uint32_t)-1;
  PlayInfoNode->data.properties = *Properties;
  PlayInfoNode->data.offset = 0;
  _PlayInfoList_linkPrev(&audio_common->PlayInfoList, audio_common->GroupList[GroupID].LastReference, PlayInfoReference);
  TH_unlock(&audio_common->PlayInfoListMutex);

  TH_lock(&audio_common->MessageQueueListMutex);
  VEC_handle0(&audio_common->MessageQueueList, 1);
  _Message_t* Message = &((_Message_t*)audio_common->MessageQueueList.ptr)[audio_common->MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::SoundPlay;
  Message->Data.SoundPlay.PlayInfoReference = PlayInfoReference;
  TH_unlock(&audio_common->MessageQueueListMutex);

  return PlayInfoReference;
}

void SoundStop(audio_t *audio, _PlayInfoList_NodeReference_t PlayInfoReference, const PropertiesSoundStop_t *Properties) {
  _audio_common_t *audio_common = (_audio_common_t *)audio;
  TH_lock(&audio_common->MessageQueueListMutex);
  VEC_handle0(&audio_common->MessageQueueList, 1);
  _Message_t* Message = &((_Message_t*)audio_common->MessageQueueList.ptr)[audio_common->MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::SoundStop;
  Message->Data.SoundStop.PlayInfoReference = PlayInfoReference;
  Message->Data.SoundStop.Properties = *Properties;
  TH_unlock(&audio_common->MessageQueueListMutex);
}

void _DataCallback(_audio_common_t *audio_common, f32_t *Output) {
  if (audio_common->MessageQueueList.Current) {
    TH_lock(&audio_common->MessageQueueListMutex);
    for (uint32_t i = 0; i < audio_common->MessageQueueList.Current; i++) {
      _Message_t* Message = &((_Message_t*)audio_common->MessageQueueList.ptr)[i];
      switch (Message->Type) {
        case _MessageType_t::SoundPlay: {
          _AddSoundToPlay(audio_common, Message->Data.SoundPlay.PlayInfoReference);
          break;
        }
        case _MessageType_t::SoundStop: {
          _RemoveFromPlayInfoList(audio_common, Message->Data.SoundStop.PlayInfoReference, &Message->Data.SoundStop.Properties);
          break;
        }
        case _MessageType_t::PauseGroup: {
          uint32_t GroupID = Message->Data.PauseGroup.GroupID;
          _PlayInfoList_NodeReference_t LastPlayInfoReference = audio_common->GroupList[GroupID].LastReference;
          _PlayInfoList_NodeReference_t PlayInfoReference = audio_common->GroupList[GroupID].FirstReference;
          TH_lock(&audio_common->PlayInfoListMutex);
          PlayInfoReference = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference)->NextNodeReference;
          TH_unlock(&audio_common->PlayInfoListMutex);
          while (PlayInfoReference != LastPlayInfoReference) {
            _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
            if (PlayInfoNode->data.PlayID != (uint32_t)-1) {
              _RemoveFromPlayList(audio_common, PlayInfoNode->data.PlayID);
              PlayInfoNode->data.PlayID = (uint32_t)-1;
            }
            PlayInfoReference = PlayInfoNode->NextNodeReference;
          }
          break;
        }
        case _MessageType_t::ResumeGroup: {
          uint32_t GroupID = Message->Data.PauseGroup.GroupID;
          _PlayInfoList_NodeReference_t LastPlayInfoReference = audio_common->GroupList[GroupID].LastReference;
          _PlayInfoList_NodeReference_t PlayInfoReference = audio_common->GroupList[GroupID].FirstReference;
          TH_lock(&audio_common->PlayInfoListMutex);
          PlayInfoReference = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference)->NextNodeReference;
          TH_unlock(&audio_common->PlayInfoListMutex);
          while (PlayInfoReference != LastPlayInfoReference) {
            _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
            if (PlayInfoNode->data.PlayID == (uint32_t)-1) {
              _AddSoundToPlay(audio_common, PlayInfoReference);
            }
            PlayInfoReference = PlayInfoNode->NextNodeReference;
          }
          break;
        }
        case _MessageType_t::StopGroup: {
          uint32_t GroupID = Message->Data.PauseGroup.GroupID;
          _PlayInfoList_NodeReference_t LastPlayInfoReference = audio_common->GroupList[GroupID].LastReference;
          _PlayInfoList_NodeReference_t PlayInfoReference = audio_common->GroupList[GroupID].FirstReference;
          TH_lock(&audio_common->PlayInfoListMutex);
          PlayInfoReference = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference)->NextNodeReference;
          while (PlayInfoReference != LastPlayInfoReference) {
            _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
            PropertiesSoundStop_t Properties;
            _RemoveFromPlayInfoList(audio_common, PlayInfoReference, &Properties);
            PlayInfoReference = PlayInfoNode->NextNodeReference;
          }
          TH_unlock(&audio_common->PlayInfoListMutex);
        }
      }
    }
    audio_common->MessageQueueList.Current = 0;
    TH_unlock(&audio_common->MessageQueueListMutex);
  }
  for (uint32_t PlayID = 0; PlayID < audio_common->PlayList.Current;) {
    _Play_t *Play = &((_Play_t *)audio_common->PlayList.ptr)[PlayID];
    _PlayInfoList_NodeReference_t PlayInfoReference = Play->Reference;
    _PlayInfoList_Node_t *PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
    piece_t *piece = PlayInfoNode->data.piece;
    PropertiesSoundPlay_t *Properties = &PlayInfoNode->data.properties;
    uint32_t OutputIndex = 0;
    uint32_t CanBeReadFrameCount;
    struct {
      f32_t FadePerFrame;
    }CalculatedVariables;
  gt_ReOffset:
    CanBeReadFrameCount = piece->GetFrameAmount() - PlayInfoNode->data.offset;
    if (CanBeReadFrameCount > _constants::CallFrameCount - OutputIndex) {
      CanBeReadFrameCount = _constants::CallFrameCount - OutputIndex;
    }
    if (Properties->Flags.FadeIn || Properties->Flags.FadeOut) {
      f32_t TotalFade = Properties->FadeTo - Properties->FadeFrom;
      f32_t CurrentFadeTime = (f32_t)CanBeReadFrameCount / _constants::opus_decode_sample_rate;
      if (TotalFade < CurrentFadeTime) {
        CanBeReadFrameCount = TotalFade * _constants::opus_decode_sample_rate;
        if (CanBeReadFrameCount == 0) {
          if(Properties->Flags.FadeIn == true){
            Properties->Flags.FadeIn = false;
          }
          else if(Properties->Flags.FadeOut == true){
            PropertiesSoundStop_t PropertiesSoundStop;
            _RemoveFromPlayInfoList(audio_common, PlayInfoReference, &PropertiesSoundStop);
            continue;
          }
        }
      }
      CalculatedVariables.FadePerFrame = (f32_t)1 / (Properties->FadeTo * _constants::opus_decode_sample_rate);
    }
    while (CanBeReadFrameCount != 0) {
      f32_t* FrameCachePointer;
      uint32_t FrameCacheAmount;
      _GetFrames(
        audio_common,
        piece,
        PlayInfoNode->data.offset,
        &FrameCachePointer,
        &FrameCacheAmount);
      if (FrameCacheAmount > CanBeReadFrameCount) {
        FrameCacheAmount = CanBeReadFrameCount;
      }
      if (Properties->Flags.FadeIn) {
        f32_t CurrentVolume = Properties->FadeFrom / Properties->FadeTo;
        for (uint32_t i = 0; i < FrameCacheAmount; i++) {
          for (uint32_t iChannel = 0; iChannel < _constants::ChannelAmount; iChannel++) {
            ((f32_t*)Output)[(OutputIndex + i) * _constants::ChannelAmount + iChannel] += FrameCachePointer[i * _constants::ChannelAmount + iChannel] * CurrentVolume;
          }
          CurrentVolume += CalculatedVariables.FadePerFrame;
        }
        Properties->FadeFrom += (f32_t)FrameCacheAmount / _constants::opus_decode_sample_rate;
      }
      else if (Properties->Flags.FadeOut) {
        f32_t CurrentVolume = Properties->FadeFrom / Properties->FadeTo;
        CurrentVolume = (f32_t)1 - CurrentVolume;
        for (uint32_t i = 0; i < FrameCacheAmount; i++) {
          for (uint32_t iChannel = 0; iChannel < _constants::ChannelAmount; iChannel++) {
            ((f32_t*)Output)[(OutputIndex + i) * _constants::ChannelAmount + iChannel] += FrameCachePointer[i * _constants::ChannelAmount + iChannel] * CurrentVolume;
          }
          CurrentVolume -= CalculatedVariables.FadePerFrame;
        }
        Properties->FadeFrom += (f32_t)FrameCacheAmount / _constants::opus_decode_sample_rate;
      }
      else {
        std::transform(
          &FrameCachePointer[0],
          &FrameCachePointer[FrameCacheAmount * _constants::ChannelAmount],
          &((f32_t*)Output)[OutputIndex * _constants::ChannelAmount],
          &((f32_t*)Output)[OutputIndex * _constants::ChannelAmount],
          std::plus<f32_t>{});
      }
      PlayInfoNode->data.offset += FrameCacheAmount;
      OutputIndex += FrameCacheAmount;
      CanBeReadFrameCount -= FrameCacheAmount;
    }

    #define JumpIfNeeded() \
      if(OutputIndex != _constants::CallFrameCount) { goto gt_ReOffset; }

    if (PlayInfoNode->data.offset == piece->GetFrameAmount()) {
      if (Properties->Flags.Loop == true) {
        PlayInfoNode->data.offset = 0;
      }
    }
    if (Properties->Flags.FadeIn) {
      if (Properties->FadeFrom >= Properties->FadeTo) {
        Properties->Flags.FadeIn = false;
      }
      JumpIfNeeded();
    }
    else if (Properties->Flags.FadeOut) {
      if (Properties->FadeFrom >= Properties->FadeTo) {
        PropertiesSoundStop_t PropertiesSoundStop;
        _RemoveFromPlayInfoList(audio_common, PlayInfoReference, &PropertiesSoundStop);
        continue;
      }
      JumpIfNeeded();
    }
    else{
      if (PlayInfoNode->data.offset == piece->GetFrameAmount()) {
        PropertiesSoundStop_t PropertiesSoundStop;
        _RemoveFromPlayInfoList(audio_common, PlayInfoReference, &PropertiesSoundStop);
        continue;
      }
      JumpIfNeeded();
    }

    #undef JumpIfNeeded

    PlayID++;
  }
}
