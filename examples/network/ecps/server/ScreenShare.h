Protocol_CI_t
ScreenShare_AddPeer
(
  ChannelList_NodeReference_t ChannelID,
  Protocol_SessionID_t SessionID,
  Protocol_ChannelSessionID_t *ResultNR
){
  auto Channel = &g_pile->ChannelList[ChannelID];
  { /* double join check */
    auto nr = Channel->SessionList.GetNodeFirst();
    while(nr != Channel->SessionList.dst){
      auto n = Channel->SessionList.GetNodeByReference(nr);
      if(n->data.SessionID == SessionID){
        *ResultNR = nr;
        return Protocol_S2C_t::JoinChannel_OK;
      }
      nr = n->NextNodeReference;
    }
  }

  auto ChannelSessionID = Channel->SessionList.NewNodeLast();
  auto ChannelSession = &Channel->SessionList[ChannelSessionID];
  ChannelSession->SessionID = SessionID;

  auto Session = &g_pile->SessionList[SessionID];

  auto SessionChannelID = Session->ChannelList.NewNodeLast();
  auto SessionChannel = &Session->ChannelList[SessionChannelID];
  SessionChannel->ChannelID = ChannelID;

  ChannelSession->SessionChannelID = SessionChannelID;
  SessionChannel->ChannelSessionID = ChannelSessionID;

  *ResultNR = ChannelSessionID;
  return Protocol_S2C_t::JoinChannel_OK;
}

bool
ScreenShare_IsSessionHost(
  Protocol_SessionID_t SessionID,
  Protocol_ChannelID_t ChannelID
){
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  return ChannelData->HostSessionID == SessionID;
}

void
ScreenShare_SessionQuit
(
  Protocol_SessionID_t SessionID,
  SessionChannelList_NodeReference_t SessionChannelID
){
  auto Session = &g_pile->SessionList[SessionID];
  if(Session->ChannelList.IsNRSentinel(SessionChannelID)){
    /* bad */
    return;
  }
  if(Session->ChannelList.inri(SessionChannelID)){
    /* bad */
    return;
  }
  if(Session->ChannelList.IsNodeReferenceRecycled(SessionChannelID)){
    /* bad */
    return;
  }
  auto SessionChannel = &Session->ChannelList[SessionChannelID];

  auto Channel = &g_pile->ChannelList[SessionChannel->ChannelID];
  ChannelSessionList_NodeReference_t ChannelSessionID = SessionChannel->ChannelSessionID;
  Session->ChannelList.unlrec(SessionChannelID);
  Channel->SessionList.unlrec(ChannelSessionID);
}

void
ScreenShare_StreamPacket
(
  Protocol_ChannelID_t ChannelID,
  void *Data,
  IO_size_t DataSize
){
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelSubData = (Channel_ScreenShare_Data_t *)Channel->Buffer;

  auto nr = Channel->SessionList.GetNodeFirst();
  ChannelSessionList_Node_t *n;
  for(; nr != Channel->SessionList.dst; nr = n->NextNodeReference){
    n = Channel->SessionList.GetNodeByReference(nr);
    if(n->data.SessionID == ChannelSubData->HostSessionID){
      /* lets dont send to self */
      continue;
    }
    auto Session = &g_pile->SessionList[n->data.SessionID];
    if(Session->UDP.Address.ip == 0){
      continue;
    }
    ProtocolUDP::S2C_t::Channel_ScreenShare_View_StreamData_t cd;
    cd.ChannelID = ChannelID;
    UDP_send(n->data.SessionID, 0, ProtocolUDP::S2C_t::Channel_ScreenShare_View_StreamData, cd, Data, DataSize);
  }
}

void
ScreenShare_SendFlagTo(
  ChannelList_NodeReference_t ChannelID,
  Protocol_SessionID_t SessionID
){
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;

  Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewSetFlag_t rest;
  rest.ChannelID = ChannelID;
  rest.Flag = ChannelData->Flag;
  Session::WriteCommand(
    SessionID,
    0,
    Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewSetFlag,
    rest);
}

void
ScreenShare_FlagIsChanged
(
  ChannelList_NodeReference_t ChannelID
){
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelSubData = (Channel_ScreenShare_Data_t *)Channel->Buffer;

  auto nr = Channel->SessionList.GetNodeFirst();
  ChannelSessionList_Node_t *n;
  for(; nr != Channel->SessionList.dst; nr = n->NextNodeReference){
    n = Channel->SessionList.GetNodeByReference(nr);
    if(n->data.SessionID == ChannelSubData->HostSessionID){
      /* lets dont send to self */
      continue;
    }
    ScreenShare_SendFlagTo(ChannelID, n->data.SessionID);
  }
}
