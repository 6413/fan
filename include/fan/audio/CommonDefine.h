void _audio_common_close(_audio_common_t *audio_common) {
  _PlayInfoList_close(&audio_common->PlayInfoList);
  A_resize(audio_common->GroupList, 0);
  VEC_free(&audio_common->PlayList);
  VEC_free(&audio_common->MessageQueueList);
  _FrameCacheList_close(&audio_common->FrameCacheList);
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

  audio_common->Tick = 0;

  _FrameCacheList_open(&audio_common->FrameCacheList);
}

void _decode_copy(
  uint8_t ChannelAmount,
  uint32_t BeginCut,
  uint32_t EndCut,
  f32_t *F2_20,
  f32_t *Output,
  uint32_t *OutputIndex
){
  switch(ChannelAmount){
    case 1:{
      /* need manuel interleave */
      for(uint32_t i = BeginCut; i < EndCut; i++){
        Output[*OutputIndex + 0] = F2_20[i];
        Output[*OutputIndex + 1] = F2_20[i];
        *OutputIndex += 2;
      }
      break;
    }
    case 2:{
      uint32_t fc = EndCut - BeginCut;
      MEM_copy(
        &F2_20[BeginCut * 2],
        &Output[*OutputIndex],
        fc * 2 * sizeof(f32_t));
      *OutputIndex += fc * 2;
      break;
    }
  }
}

void _decode(piece_t *piece, f32_t *Output, uint64_t offset, uint32_t FrameCount) {
  uint64_t PieceFrameOffset = offset;

  uint32_t SegmentIndex = PieceFrameOffset / _constants::Opus::SegmentFrameAmount20;

  uint32_t OutputIndex = 0;
  while(OutputIndex / 2 != FrameCount){
    uint32_t BeginCut = 0;
    if(offset > SegmentIndex * _constants::Opus::SegmentFrameAmount20){
      BeginCut = (SegmentIndex * _constants::Opus::SegmentFrameAmount20) % _constants::Opus::SegmentFrameAmount20;
    }
    uint32_t EndCut = _constants::Opus::SegmentFrameAmount20;
    if(OutputIndex / 2 + _constants::Opus::SegmentFrameAmount20 > FrameCount){
      EndCut = (OutputIndex / 2 + _constants::Opus::SegmentFrameAmount20) % _constants::Opus::SegmentFrameAmount20;
    }

    _SACSegment_t *SACSegment = &((_SACSegment_t *)piece->SACData)[SegmentIndex];
    f32_t F2_20[_constants::Opus::SegmentFrameAmount20 * 5 * 2];
    sint32_t oerr = opus_decode_float(
      piece->od,
      &piece->SACData[SACSegment->Offset],
      SACSegment->Size,
      F2_20,
      _constants::Opus::SegmentFrameAmount20 * 5,
      0);

    if(oerr != _constants::Opus::SegmentFrameAmount20){
      fan::print("help", oerr);
      fan::throw_error("a");
    }

    _decode_copy(piece->ChannelAmount, BeginCut, EndCut, F2_20, Output, &OutputIndex);

    SegmentIndex++;
  }
}

void _GetFrames(_FrameCacheList_t *FrameCacheList, piece_t *piece, uint64_t Offset, uint64_t Time, f32_t **FramePointer, uint32_t *FrameAmount) {
  uint32_t PieceCacheIndex = Offset / _constants::FrameCacheAmount;
  _PieceCache_t* PieceCache = &piece->Cache[PieceCacheIndex];
  if (PieceCache->ref == (_FrameCacheList_NodeReference_t)-1) {
    PieceCache->ref = _FrameCacheList_NewNodeLast(FrameCacheList);
    _FrameCacheList_Node_t* FrameCacheList_Node = _FrameCacheList_GetNodeByReference(FrameCacheList, PieceCache->ref);
    uint64_t FrameOffset = (uint64_t)PieceCacheIndex * _constants::FrameCacheAmount;
    uint32_t FrameAmount = _constants::FrameCacheAmount;
    if (FrameOffset + FrameAmount > piece->FrameAmount) {
      FrameAmount = piece->FrameAmount - FrameOffset;
    }
    _decode(piece, &FrameCacheList_Node->data.Frames[0][0], FrameOffset, FrameAmount);
    FrameCacheList_Node->data.piece = piece;
    FrameCacheList_Node->data.PieceCacheIndex = PieceCacheIndex;
  }
  else {
    _FrameCacheList_Unlink(FrameCacheList, PieceCache->ref);
    _FrameCacheList_linkPrev(FrameCacheList, FrameCacheList->dst, PieceCache->ref);
  }
  _FrameCacheList_Node_t* FrameCacheList_Node = _FrameCacheList_GetNodeByReference(FrameCacheList, PieceCache->ref);
  FrameCacheList_Node->data.LastAccessTime = Time;
  *FramePointer = &FrameCacheList_Node->data.Frames[Offset % _constants::FrameCacheAmount][0];
  *FrameAmount = _constants::FrameCacheAmount - Offset % _constants::FrameCacheAmount;
}

void _AddSoundToPlay(_audio_common_t *audio_common, _PlayInfoList_NodeReference_t PlayInfoReference) {
  VEC_handle0(&audio_common->PlayList, 1);
  uint32_t PlayID = audio_common->PlayList.Current - 1;
  _Play_t *Play = &((_Play_t *)audio_common->PlayList.ptr)[PlayID];
  Play->Reference = PlayInfoReference;
  _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
  #if fan_debug >= 0
    if (PlayInfoNode->data.PlayID != (uint32_t)-1) {
      /* trying play sound that already playing */
      fan::throw_error("fan_debug");
    }
  #endif
  PlayInfoNode->data.PlayID = PlayID;
}
void _RemoveFromPlayList(_audio_common_t *audio_common, uint32_t PlayID) {
  /* super fast remove */
  ((_Play_t *)audio_common->PlayList.ptr)[PlayID] = ((_Play_t *)audio_common->PlayList.ptr)[--audio_common->PlayList.Current];

  /* moved one needs update */
  _PlayInfoList_NodeReference_t PlayInfoReference = ((_Play_t *)audio_common->PlayList.ptr)[PlayID].Reference;
  _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
  PlayInfoNode->data.PlayID = PlayID;
}
void _RemoveFromPlayInfoList(_audio_common_t *audio_common, uint32_t PlayInfoReference, const PropertiesSoundStop_t* Properties) {
  _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
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

  piece->ChannelAmount = SACHead->ChannelAmount;

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
      DataOffset += BeforeSum;
      BeforeSum = 0;
      psi++;
    }
  }

  #undef tbs

  {
    int err;
    piece->od = opus_decoder_create(48000, piece->ChannelAmount, &err);
    if(err != OPUS_OK){
      fan::throw_error("why");
    }
  }

  piece->FrameOffset = 0;

  piece->FrameAmount = (uint64_t)piece->TotalSegments * _constants::Opus::SegmentFrameAmount20;

  uint32_t CacheAmount = piece->FrameAmount / _constants::FrameCacheAmount + !!(piece->FrameAmount % _constants::FrameCacheAmount);
  piece->Cache = (_PieceCache_t*)A_resize(0, CacheAmount * sizeof(_PieceCache_t));
  for (uint32_t i = 0; i < CacheAmount; i++) {
    piece->Cache[i].ref = (_FrameCacheList_NodeReference_t)-1;
  }

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

uint32_t SoundPlay(audio_t *audio, piece_t *piece, uint32_t GroupID, const PropertiesSoundPlay_t *Properties) {
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
    _PlayInfoList_Node_t* PlayInfoNode = _PlayInfoList_GetNodeByReference(&audio_common->PlayInfoList, PlayInfoReference);
    piece_t* piece = PlayInfoNode->data.piece;
    PropertiesSoundPlay_t *Properties = &PlayInfoNode->data.properties;
    uint32_t OutputIndex = 0;
    uint32_t CanBeReadFrameCount;
    struct {
      f32_t FadePerFrame;
    }CalculatedVariables;
  gt_ReOffset:
    CanBeReadFrameCount = piece->FrameAmount - PlayInfoNode->data.offset;
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
        &audio_common->FrameCacheList,
        piece,
        PlayInfoNode->data.offset,
        audio_common->Tick,
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

    if (PlayInfoNode->data.offset == piece->FrameAmount) {
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
      if (PlayInfoNode->data.offset == piece->FrameAmount) {
        PropertiesSoundStop_t PropertiesSoundStop;
        _RemoveFromPlayInfoList(audio_common, PlayInfoReference, &PropertiesSoundStop);
        continue;
      }
      JumpIfNeeded();
    }

    #undef JumpIfNeeded

    PlayID++;
  }
  for (_FrameCacheList_NodeReference_t ref = _FrameCacheList_GetNodeFirst(&audio_common->FrameCacheList); ref != audio_common->FrameCacheList.dst;) {
    _FrameCacheList_Node_t* node = _FrameCacheList_GetNodeByReference(&audio_common->FrameCacheList, ref);
    if (node->data.LastAccessTime + _constants::FrameCacheTime > audio_common->Tick) {
      break;
    }
    node->data.piece->Cache[node->data.PieceCacheIndex].ref = (_FrameCacheList_NodeReference_t)-1;
    _FrameCacheList_NodeReference_t NextReference = node->NextNodeReference;
    _FrameCacheList_Unlink(&audio_common->FrameCacheList, ref);
    _FrameCacheList_Recycle(&audio_common->FrameCacheList, ref);
    ref = NextReference;
  }
  audio_common->Tick++;
}
