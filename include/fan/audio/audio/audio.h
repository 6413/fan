using piece_t = system_audio_t::piece_t;
using SoundPlayID_t = system_audio_t::SoundPlayID_t;
using PropertiesSoundPlay_t = system_audio_t::PropertiesSoundPlay_t;
using PropertiesSoundStop_t = system_audio_t::PropertiesSoundStop_t;
using _SACHead_t = system_audio_t::_SACHead_t;
using _SACSegment_t = system_audio_t::_SACSegment_t;
using _Message_t = system_audio_t::_Message_t;
using _MessageType_t = system_audio_t::_MessageType_t;

system_audio_t *system_audio;

void bind(system_audio_t *p_system_audio){
  system_audio = p_system_audio;
}
void unbind(){

}

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
      ((_SACSegment_t *)piece->SACData)[psi].CacheID = system_audio_t::_CacheList_gnric();
      DataOffset += BeforeSum;
      BeforeSum = 0;
      psi++;
    }
  }

  #undef tbs

  piece->FrameAmount = (uint64_t)piece->TotalSegments * system_audio_t::_constants::Opus::SegmentFrameAmount20;
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

  err = Open(piece, sd.data(), sd.size());
  if(err != 0){
    return err;
  }

  return 0;
}
void Close(piece_t *piece){
  A_resize(piece->SACData, 0);
}

SoundPlayID_t SoundPlay(piece_t *piece, const PropertiesSoundPlay_t *Properties) {
  if (Properties->GroupID >= system_audio->Process.GroupAmount) {
    TH_lock(&system_audio->Process.PlayInfoListMutex);
    system_audio->Process.GroupList = (system_audio_t::Process_t::_Group_t *)A_resize(system_audio->Process.GroupList, sizeof(system_audio_t::Process_t::_Group_t) * (Properties->GroupID + 1));
    for(; system_audio->Process.GroupAmount <= Properties->GroupID; ++system_audio->Process.GroupAmount){
      system_audio->Process.GroupList[system_audio->Process.GroupAmount].FirstReference = system_audio->Process.PlayInfoList.NewNodeLast_alloc();
      system_audio->Process.GroupList[system_audio->Process.GroupAmount].LastReference = system_audio->Process.PlayInfoList.NewNodeLast_alloc();
    }
  }
  else{
    TH_lock(&system_audio->Process.PlayInfoListMutex);
  }
  auto pnr = system_audio->Process.PlayInfoList.NewNode();
  auto PlayInfo = &system_audio->Process.PlayInfoList[pnr];
  PlayInfo->piece = piece;
  PlayInfo->GroupID = Properties->GroupID;
  PlayInfo->PlayID = (uint32_t)-1;
  PlayInfo->properties = *Properties;
  PlayInfo->offset = 0;
  PlayInfo->unique = system_audio->Process.PlayInfoListUnique++;
  system_audio->Process.PlayInfoList.linkPrev(system_audio->Process.GroupList[Properties->GroupID].LastReference, pnr);
  TH_unlock(&system_audio->Process.PlayInfoListMutex);

  TH_lock(&system_audio->Process.MessageQueueListMutex);
  VEC_handle0(&system_audio->Process.MessageQueueList, 1);
  _Message_t* Message = &((_Message_t *)system_audio->Process.MessageQueueList.ptr)[system_audio->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::SoundPlay;
  Message->Data.SoundPlay.PlayInfoReference = pnr;
  TH_unlock(&system_audio->Process.MessageQueueListMutex);

  return {.nr = pnr, .unique = PlayInfo->unique};
}
void SoundStop(SoundPlayID_t &SoundPlayID, const PropertiesSoundStop_t *Properties) {
  TH_lock(&system_audio->Process.MessageQueueListMutex);
  VEC_handle0(&system_audio->Process.MessageQueueList, 1);
  _Message_t* Message = &((_Message_t *)system_audio->Process.MessageQueueList.ptr)[system_audio->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::SoundStop;
  Message->Data.SoundStop.SoundPlayID = SoundPlayID;
  Message->Data.SoundStop.Properties = *Properties;
  TH_unlock(&system_audio->Process.MessageQueueListMutex);
}

void PauseGroup(uint32_t GroupID) {
  TH_lock(&system_audio->Process.MessageQueueListMutex);
  VEC_handle0(&system_audio->Process.MessageQueueList, 1);
  _Message_t *Message = &((_Message_t *)system_audio->Process.MessageQueueList.ptr)[system_audio->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::PauseGroup;
  Message->Data.PauseGroup.GroupID = GroupID;
  TH_unlock(&system_audio->Process.MessageQueueListMutex);
}
void ResumeGroup(uint32_t GroupID) {
  TH_lock(&system_audio->Process.MessageQueueListMutex);
  VEC_handle0(&system_audio->Process.MessageQueueList, 1);
  _Message_t *Message = &((_Message_t*)system_audio->Process.MessageQueueList.ptr)[system_audio->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::ResumeGroup;
  Message->Data.ResumeGroup.GroupID = GroupID;
  TH_unlock(&system_audio->Process.MessageQueueListMutex);
}

void StopGroup(uint32_t GroupID) {

}

void SetVolume(f32_t Volume) {
  system_audio->Out.SetVolume(Volume);
}
f32_t GetVolume() {
  return system_audio->Out.GetVolume();
}
