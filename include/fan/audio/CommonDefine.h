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

void _seek(_OpusHolder_t* holder, uint64_t offset) {
  sint32_t err = op_pcm_seek(holder->decoder, offset);
  if (err != 0) {
    // TOOD
    // op_pcm_seek() documented as seeking/reading from file.
    // in our design we dont want to do file i/o at realtime.
    // so if seek fails for some reason it must be fatal problem.
    fan::throw_error("fan::audio::seek failed to seek audio file");
  }
}

void _decode(_OpusHolder_t* holder, f32_t* Output, uint64_t offset, uint32_t FrameCount) {
  _seek(holder, offset);

  uint64_t TotalRead = 0;

  while (TotalRead != FrameCount) {
    int read = op_read_float_stereo(
      holder->decoder,
      &Output[TotalRead * _constants::ChannelAmount],
      (FrameCount - TotalRead) * _constants::ChannelAmount);
    if (read <= 0) {
      fan::throw_error("help " + ::std::to_string(read));
      break;
    }

    TotalRead += read;
  }
}

void _GetFrames(_FrameCacheList_t *FrameCacheList, piece_t* piece, uint64_t Offset, uint64_t Time, f32_t** FramePointer, uint32_t* FrameAmount) {
  uint32_t PieceCacheIndex = Offset / _constants::FrameCacheAmount;
  _PieceCache_t* PieceCache = &piece->Cache[PieceCacheIndex];
  if (PieceCache->ref == (_FrameCacheList_NodeReference_t)-1) {
    PieceCache->ref = _FrameCacheList_NewNodeLast(FrameCacheList);
    _FrameCacheList_Node_t* FrameCacheList_Node = _FrameCacheList_GetNodeByReference(FrameCacheList, PieceCache->ref);
    uint64_t FrameOffset = (uint64_t)PieceCacheIndex * _constants::FrameCacheAmount;
    uint32_t FrameAmount = _constants::FrameCacheAmount;
    if (FrameOffset + FrameAmount > piece->raw_size) {
      FrameAmount = piece->raw_size - FrameOffset;
    }
    _decode(&piece->holder, &FrameCacheList_Node->data.Frames[0][0], FrameOffset, FrameAmount);
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

void _piece_open_rest(_audio_common_t *audio_common, fan::audio::piece_t *piece){
  piece->audio_common = audio_common;
  piece->raw_size = op_pcm_total(piece->holder.decoder, 0);
  uint32_t CacheAmount = piece->raw_size / _constants::FrameCacheAmount + !!(piece->raw_size % _constants::FrameCacheAmount);
  piece->Cache = (_PieceCache_t*)A_resize(0, CacheAmount * sizeof(_PieceCache_t));
  for (uint32_t i = 0; i < CacheAmount; i++) {
    piece->Cache[i].ref = (_FrameCacheList_NodeReference_t)-1;
  }
}
sint32_t piece_open(audio_t *audio, fan::audio::piece_t* piece, const ::std::string& path) {
  sint32_t err;
  piece->holder.decoder = op_open_file(path.c_str(), &err);
  if (err != 0) {
    return err;
  }
  piece->holder.head = op_head(piece->holder.decoder, 0);

  _piece_open_rest((_audio_common_t *)audio, piece);

  return 0;
}
sint32_t piece_open(audio_t *audio, fan::audio::piece_t* piece, void *data, uintptr_t size) {
  sint32_t err;
  piece->holder.decoder = op_open_memory((const uint8_t *)data, size, &err);
  if (err != 0) {
    return err;
  }
  piece->holder.head = op_head(piece->holder.decoder, 0);

  _piece_open_rest((_audio_common_t *)audio, piece);

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
    CanBeReadFrameCount = piece->raw_size - PlayInfoNode->data.offset;
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

    if (PlayInfoNode->data.offset == piece->raw_size) {
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
      if (PlayInfoNode->data.offset == piece->raw_size) {
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
