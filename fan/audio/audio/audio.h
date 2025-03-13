using piece_t = system_audio_t::piece_t;
using SoundPlayID_t = system_audio_t::SoundPlayID_t;
using PropertiesSoundPlay_t = system_audio_t::PropertiesSoundPlay_t;
using PropertiesSoundStop_t = system_audio_t::PropertiesSoundStop_t;
using _SACHead_t = system_audio_t::_SACHead_t;
using _SACSegment_t = system_audio_t::_SACSegment_t;
using _Message_t = system_audio_t::_Message_t;
using _MessageType_t = system_audio_t::_MessageType_t;
using PieceFlag = system_audio_t::PieceFlag;

system_audio_t *system_audio;

void bind(system_audio_t *p_system_audio){
  system_audio = p_system_audio;
}
void unbind(){

}

sint32_t Open(piece_t *piece, FS_file_t *file, PieceFlag::t Flag){
  IO_fd_t file_fd;
  FS_file_getfd(file, &file_fd);

  sint32_t err = 0;

  _SACHead_t SACHead;
  if(FS_file_read(file, &SACHead, sizeof(_SACHead_t)) != sizeof(_SACHead_t)){
   err = -1;
   goto gt_r0_0;
  }

  if(SACHead.Sign != 0xff){
   err = -1;
   goto gt_r0_0;
  }

  piece->_piece = new system_audio_t::_piece_t;
  system_audio_t::_piece_t  *_piece; _piece = piece->_piece;

  _piece->ChannelAmount = SACHead.ChannelAmount;
  _piece->BeginCut = SACHead.BeginCut;

  uint8_t *SACSegmentSizes; SACSegmentSizes = (uint8_t *)A_resize(0, SACHead.TotalSegments);
  if(FS_file_read(file, SACSegmentSizes, SACHead.TotalSegments) != SACHead.TotalSegments){
   err = -1;
   goto gt_r0_2;
  }

  _piece->TotalSegments = 0;
  uint64_t TotalOfSACSegmentSizes; TotalOfSACSegmentSizes = 0;
  for(uint32_t i = 0; i < SACHead.TotalSegments; i++){
   TotalOfSACSegmentSizes += SACSegmentSizes[i];
   if(SACSegmentSizes[i] == 0xff){
     continue;
   }
   _piece->TotalSegments++;
  }

  _piece->FrameAmount = (uint64_t)_piece->TotalSegments * system_audio_t::_constants::Opus::SegmentFrameAmount20;
  if(SACHead.EndCut >= _piece->FrameAmount){
   err = -1;
   goto gt_r0_2;
  }
  _piece->FrameAmount -= SACHead.EndCut;

  IO_stat_t s;
  err = IO_fstat(&file_fd, &s);
  if(err != 0){
   goto gt_r0_2;
  }

  IO_off_t TotalFileSize; TotalFileSize = IO_stat_GetSizeInBytes(&s);
  IO_off_t FileIsAt; FileIsAt = sizeof(_SACHead_t) + SACHead.TotalSegments;

  uint64_t LeftSize; LeftSize = TotalFileSize - FileIsAt;
  if(TotalOfSACSegmentSizes != LeftSize){
   err = -1;
   goto gt_r0_2;
  }

  if(Flag & PieceFlag::nonsimu){
   _piece->StoreType = system_audio_t::_piece_t::StoreType_t::nonsimu;
   _piece->StoreData.nonsimu.m = FileIsAt % PAGE_SIZE;

   uintptr_t MapAt = FileIsAt - _piece->StoreData.nonsimu.m;
   uint64_t MapLeftSize = TotalFileSize - MapAt;

   //sintptr_t ptr = IO_mmap(NULL, MapLeftSize, PROT_READ, MAP_SHARED, file_fd.fd, MapAt);
   //printf("pointer is %ld %lu %lu\n", ptr, FileIsAt, MapLeftSize);
   //if(ptr < 0){
   //  err = ptr;
   //  goto gt_r0_2;
   //}
   //_piece->SACData = (uint8_t *)(ptr + _piece->StoreData.nonsimu.m);
  }
  else{
   _piece->StoreType = system_audio_t::_piece_t::StoreType_t::normal;

   _piece->SACData = A_resize(0, LeftSize);
   if(FS_file_read(file, _piece->SACData, LeftSize) != LeftSize){
     err = -1;
     goto gt_r0_3;
   }
  }

  {
   uint16_t BeforeSum = 0;
   uint32_t psi = 0;
   uint32_t DataOffset = 0;
   _piece->SACSegment = (_SACSegment_t *)A_resize(0, _piece->TotalSegments * sizeof(_SACSegment_t));
   for(uint32_t i = 0; i < SACHead.TotalSegments; i++){
     BeforeSum += SACSegmentSizes[i];
     if(SACSegmentSizes[i] == 0xff){
       continue;
     }
     _piece->SACSegment[psi].Offset = DataOffset;
     _piece->SACSegment[psi].Size = BeforeSum;
     _piece->SACSegment[psi].CacheID.sic();
     DataOffset += BeforeSum;
     BeforeSum = 0;
     psi++;
   }
  }

  A_resize(SACSegmentSizes, 0);
  return 0;

  A_resize(_piece->SACSegment, 0);
  gt_r0_3:
  if(Flag & PieceFlag::nonsimu){
   // IO_unmap() TODO
  }
  else{
   A_resize(_piece->SACData, 0);
  }
  gt_r0_2:
  A_resize(SACSegmentSizes, 0);
  delete _piece;
  gt_r0_0:
  return err;
}

sint32_t Open(piece_t *piece, const std::string &path, uint32_t Flag) {
  sint32_t err;

  FS_file_t file;
  err = FS_file_open(path.c_str(), &file, O_RDONLY);
  if(err != 0){
    return err;
  }

  err = Open(piece, &file, Flag);

  FS_file_close(&file);

  return err;
}
void Close(piece_t *piece){
  TH_lock(&system_audio->Process.MessageQueueListMutex);
  VEC_handle0(&system_audio->Process.MessageQueueList, 1);
  _Message_t* Message = &((_Message_t *)system_audio->Process.MessageQueueList.ptr)[system_audio->Process.MessageQueueList.Current - 1];
  Message->Type = _MessageType_t::ClosePiece;
  Message->Data.ClosePiece._piece = piece->_piece;
  TH_unlock(&system_audio->Process.MessageQueueListMutex);
}

SoundPlayID_t SoundPlay(piece_t *piece, const PropertiesSoundPlay_t *Properties) {
  if(Properties->GroupID >= system_audio->Process.GroupAmount) {
    TH_lock(&system_audio->Process.PlayInfoListMutex);
    system_audio->Process.GroupList = (system_audio_t::Process_t::_Group_t *)A_resize(system_audio->Process.GroupList, sizeof(system_audio_t::Process_t::_Group_t) * (Properties->GroupID + 1));
    for(; system_audio->Process.GroupAmount <= Properties->GroupID; ++system_audio->Process.GroupAmount){
      system_audio->Process.GroupList[system_audio->Process.GroupAmount].FirstReference = system_audio->Process.PlayInfoList.NewNodeLast();
      system_audio->Process.GroupList[system_audio->Process.GroupAmount].LastReference = system_audio->Process.PlayInfoList.NewNodeLast();
    }
  }
  else{
    TH_lock(&system_audio->Process.PlayInfoListMutex);
  }
  auto pnr = system_audio->Process.PlayInfoList.NewNode();
  auto PlayInfo = &system_audio->Process.PlayInfoList[pnr];
  PlayInfo->_piece = piece->_piece;
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
