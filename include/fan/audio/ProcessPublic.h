sint32_t Open(piece_t *piece, const void *data, uintptr_t size) {
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
      ((_SACSegment_t *)piece->SACData)[psi].CacheID = _CacheList_gnric();
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
sint32_t Open(piece_t *piece, const fan::string &path) {
  sint32_t err;

  fan::string sd;
  err = fan::io::file::read(path, &sd);
  if(err != 0){
    return err;
  }

  err = this->Open(piece, sd.data(), sd.size());
  if(err != 0){
    return err;
  }

  return 0;
}
void Close(piece_t *piece){
  A_resize(piece->SACData, 0);
}

SoundPlayID_t SoundPlay(piece_t *piece, const PropertiesSoundPlay_t *Properties) {
  if (Properties->GroupID >= this->Process.GroupAmount) {
    TH_lock(&this->Process.PlayInfoListMutex);
    this->Process.GroupList = (Process_t::_Group_t *)A_resize(this->Process.GroupList, sizeof(Process_t::_Group_t) * Properties->GroupID);
    for(; this->Process.GroupAmount <= Properties->GroupID; ++this->Process.GroupAmount){
      this->Process.GroupList[this->Process.GroupAmount].FirstReference = this->Process.PlayInfoList.NewNodeLast_alloc();
      this->Process.GroupList[this->Process.GroupAmount].LastReference = this->Process.PlayInfoList.NewNodeLast_alloc();
    }
  }
  else{
    TH_lock(&this->Process.PlayInfoListMutex);
  }
  _PlayInfoList_NodeReference_t PlayInfoReference = this->Process.PlayInfoList.NewNode();
  auto PlayInfo = &this->Process.PlayInfoList[PlayInfoReference];
  PlayInfo->piece = piece;
  PlayInfo->GroupID = Properties->GroupID;
  PlayInfo->PlayID = (uint32_t)-1;
  PlayInfo->properties = *Properties;
  PlayInfo->offset = 0;
  this->Process.PlayInfoList.linkPrev(this->Process.GroupList[Properties->GroupID].LastReference, PlayInfoReference);
  TH_unlock(&this->Process.PlayInfoListMutex);

  TH_lock(&this->Process.MessageQueueListMutex);
  VEC_handle0(&this->Process.MessageQueueList, 1);
  _Message_t* Message = &((_Message_t *)this->Process.MessageQueueList.ptr)[this->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::SoundPlay;
  Message->Data.SoundPlay.PlayInfoReference = PlayInfoReference;
  TH_unlock(&this->Process.MessageQueueListMutex);

  return PlayInfoReference;
}
void SoundStop(_PlayInfoList_NodeReference_t PlayInfoReference, const PropertiesSoundStop_t *Properties) {
  TH_lock(&this->Process.MessageQueueListMutex);
  VEC_handle0(&this->Process.MessageQueueList, 1);
  _Message_t* Message = &((_Message_t *)this->Process.MessageQueueList.ptr)[this->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::SoundStop;
  Message->Data.SoundStop.PlayInfoReference = PlayInfoReference;
  Message->Data.SoundStop.Properties = *Properties;
  TH_unlock(&this->Process.MessageQueueListMutex);
}

void PauseGroup(uint32_t GroupID) {
  TH_lock(&this->Process.MessageQueueListMutex);
  VEC_handle0(&this->Process.MessageQueueList, 1);
  _Message_t *Message = &((_Message_t *)this->Process.MessageQueueList.ptr)[this->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::PauseGroup;
  Message->Data.PauseGroup.GroupID = GroupID;
  TH_unlock(&this->Process.MessageQueueListMutex);
}
void ResumeGroup(uint32_t GroupID) {
  TH_lock(&this->Process.MessageQueueListMutex);
  VEC_handle0(&this->Process.MessageQueueList, 1);
  _Message_t *Message = &((_Message_t*)this->Process.MessageQueueList.ptr)[this->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::ResumeGroup;
  Message->Data.ResumeGroup.GroupID = GroupID;
  TH_unlock(&this->Process.MessageQueueListMutex);
}

void StopGroup(uint32_t GroupID) {

}
