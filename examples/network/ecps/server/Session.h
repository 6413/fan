namespace Session{

bool IsInvalid(Protocol_SessionID_t SessionID){
  if(g_pile->SessionList.inri(SessionID)){
    return true;
  }
  if(g_pile->SessionList.IsNRSentinel(SessionID)){
    return true;
  }
  if(g_pile->SessionList.IsNodeReferenceRecycled(SessionID)){
    return true;
  }

  return false;
}

bool IsIdentifyInvalid(Protocol_SessionID_t SessionID, uint64_t IdentifySecret){
  auto Session = &g_pile->SessionList[SessionID];
  if(Session->UDP.IdentifySecret != IdentifySecret){
    return true;
  }
  return false;
}

void WriteCommand(Protocol_SessionID_t SessionID, uint32_t ID, Protocol_CI_t Command, auto&&...args){
  auto Session = &g_pile->SessionList[SessionID];
  TCP_WriteCommand(Session->TCP.peer, ID, Command, args...);
}

Protocol_SessionID_t OpenFromTCPPeer(NET_TCP_peer_t *peer){
  auto SessionID = g_pile->SessionList.NewNodeLast();
  auto Session = &g_pile->SessionList[SessionID];

  Session->AccountID = Protocol_AccountID_t::GetInvalid();

  Session->ChannelList.Open();

  Session->TCP.peer = peer;

  Session->UDP.LastInvalidIdentifyAt = 0;
  Session->UDP.Address.ip = 0;
  Session->UDP.IdentifySecret = RAND_hard_64();

  return SessionID;
}

void _Close_ChannelClose(Protocol_ChannelID_t ChannelID){
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto nr = Channel->SessionList.GetNodeFirst();
  while(nr != Channel->SessionList.dst){
    auto n = Channel->SessionList.GetNodeByReference(nr);
    auto _nr = n->NextNodeReference;
    Channel_KickSession(
      ChannelID,
      nr,
      Protocol::KickedFromChannel_Reason_t::ChannelIsClosed);
    nr = _nr;
  }
}

void Close(Protocol_SessionID_t SessionID){
  auto Session = &g_pile->SessionList[SessionID];

  auto nr = Session->ChannelList.GetNodeFirst();
  while(nr != Session->ChannelList.dst){
    auto n = Session->ChannelList.GetNodeByReference(nr);
    Protocol_ChannelID_t ChannelID = n->data.ChannelID;
    auto Channel = &g_pile->ChannelList[ChannelID];
    Channel->SessionList.unlrec(n->data.ChannelSessionID);
    switch(Channel->Type){
      case Protocol::ChannelType_ScreenShare_e:{
        auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
        if(ChannelData->HostSessionID == SessionID){
          _Close_ChannelClose(ChannelID);
          g_pile->ChannelList.unlrec(ChannelID);
          goto gt_NextChannel;
        }
        break;
      }
    }
    gt_NextChannel:;
    nr = n->NextNodeReference;
  }
  Session->ChannelList.Close();
}

} // namespace Session
