case Protocol_C2S_t::KeepAlive:{
  TCP_WriteCommand(
    peer,
    BasePacket->ID,
    Protocol_S2C_t::KeepAlive);

  goto StateDone_gt;
}
case Protocol_C2S_t::Request_Login:{
  auto Request = (Protocol_C2S_t::Request_Login_t *)RestPacket;

  if(Session->AccountID != Protocol_AccountID_t::GetInvalid()){
    /* double login */
    goto StateDone_gt;
  }

  switch(Request->Type){
    case Protocol::LoginType_t::Anonymous:{
      // todo give account
      break;
    }
    default:{
      goto StateDone_gt;
    }
  }

  Protocol_S2C_t::Response_Login_t rest;
  rest.AccountID = Session->AccountID;
  rest.SessionID = SessionID;

  TCP_WriteCommand(
    peer,
    BasePacket->ID,
    Protocol_S2C_t::Response_Login,
    rest);

  goto StateDone_gt;
}
case Protocol_C2S_t::CreateChannel:{
  auto Request = (Protocol_C2S_t::CreateChannel_t *)RestPacket;
  if(Request->Type >= Protocol::ChannelType_Amount){
    Protocol_S2C_t::JoinChannel_Error_t rest;
    rest.Reason = Protocol::JoinChannel_Error_Reason_t::InvalidChannelType;
    TCP_WriteCommand(
      peer,
      BasePacket->ID,
      Protocol_S2C_t::CreateChannel_Error,
      rest);
    goto StateDone_gt;
  }

  Protocol_ChannelID_t ChannelID;
  Protocol_ChannelSessionID_t ChannelSessionID;
  switch(Request->Type){
    case Protocol::ChannelType_ScreenShare_e:{
      ChannelID = AddChannel_ScreenShare(SessionID);

      Protocol_CI_t CI = ScreenShare_AddPeer(
        ChannelID,
        SessionID,
        &ChannelSessionID);
      if(CI != Protocol_S2C_t::JoinChannel_OK){
        /* why */
        __abort();
      }

      break;
    }
    default:{
      __abort();
    }
  }

  {
    Protocol_S2C_t::JoinChannel_OK_t jc;
    jc.Type = Protocol::ChannelType_ScreenShare_e;
    jc.ChannelID = ChannelID;
    jc.ChannelSessionID = ChannelSessionID;

    TCP_WriteCommand(
      peer,
      BasePacket->ID,
      Protocol_S2C_t::CreateChannel_OK,
      jc);
  }

  goto StateDone_gt;
}
case Protocol_C2S_t::JoinChannel:{

  auto Request = (Protocol_C2S_t::JoinChannel_t *)RestPacket;
  Protocol_ChannelID_t ChannelID = Request->ChannelID;

  if(IsChannelInvalid(ChannelID) == true){
    Protocol_S2C_t::JoinChannel_Error_t rest;
    rest.Reason = Protocol::JoinChannel_Error_Reason_t::InvalidChannelID;
    TCP_WriteCommand(
      peer,
      BasePacket->ID,
      Protocol_S2C_t::JoinChannel_Error,
      rest);
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];

  Protocol_ChannelSessionID_t ChannelSessionID;
  switch(Channel->Type){
    case Protocol::ChannelType_ScreenShare_e:{
      Protocol_CI_t CI = ScreenShare_AddPeer(
        ChannelID,
        SessionID,
        &ChannelSessionID);
      if(CI != Protocol_S2C_t::JoinChannel_OK){
        TCP_WriteCommand(peer, BasePacket->ID, CI);
        goto StateDone_gt;
      }

      break;
    }
  }

  {
    Protocol_S2C_t::JoinChannel_OK_t jc;
    jc.Type = Protocol::ChannelType_ScreenShare_e;
    jc.ChannelID = ChannelID;
    jc.ChannelSessionID = ChannelSessionID;
    TCP_WriteCommand(
      peer,
      BasePacket->ID,
      Protocol_S2C_t::JoinChannel_OK,
      jc);
  }

  ScreenShare_SendFlagTo(ChannelID, {ChannelID, ChannelSessionID});

  goto StateDone_gt;
}
case Protocol_C2S_t::QuitChannel:{
  /* not implemented yet */
  __abort();
  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_Share_InformationToViewSetFlag:{

  auto Request = (Protocol_C2S_t::Channel_ScreenShare_Share_InformationToViewSetFlag_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  if(ChannelData->HostSessionID != SessionID){
    goto StateDone_gt;
  }
  ChannelData->Flag = Request->Flag;
  ScreenShare_FlagIsChanged(ChannelID);

  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_Share_InformationToViewMouseCoordinate:{

  auto Request = (Protocol_C2S_t::Channel_ScreenShare_Share_InformationToViewMouseCoordinate_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  if(ChannelData->HostSessionID != SessionID){
    goto StateDone_gt;
  }

  auto nr = Channel->SessionList.GetNodeFirst();
  ChannelSessionList_Node_t *n;
  for(; nr != Channel->SessionList.dst; nr = n->NextNodeReference){
    n = Channel->SessionList.GetNodeByReference(nr);
    if(n->data.SessionID == SessionID){
      /* lets dont send to self */
      continue;
    }

    Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewMouseCoordinate_t Payload;
    Payload.ChannelID = ChannelID;
    Payload.pos = Request->pos;

    Session::WriteCommand(
      n->data.SessionID,
      0,
      Protocol_S2C_t::Channel_ScreenShare_View_InformationToViewMouseCoordinate,
      Payload);
  }

  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostMouseCoordinate:{

  auto Request = (Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostMouseCoordinate_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  if(!(ChannelData->Flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl)){
    goto StateDone_gt;
  }

  Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseCoordinate_t Payload;
  Payload.ChannelID = ChannelID;
  Payload.pos = Request->pos;

  Session::WriteCommand(
    ChannelData->HostSessionID,
    0,
    Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseCoordinate,
    Payload);

  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostMouseMotion:{

  auto Request = (Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostMouseMotion_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  if(!(ChannelData->Flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl)){
    goto StateDone_gt;
  }

  Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseMotion_t Payload;
  Payload.ChannelID = ChannelID;
  Payload.Motion = Request->Motion;

  Session::WriteCommand(
    ChannelData->HostSessionID,
    0,
    Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseMotion,
    Payload);

  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostMouseButton:{

  auto Request = (Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostMouseButton_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  if(!(ChannelData->Flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl)){
    goto StateDone_gt;
  }

  Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseButton_t Payload;
  Payload.ChannelID = ChannelID;
  Payload.key = Request->key;
  Payload.state = Request->state;
  Payload.pos = Request->pos;

  Session::WriteCommand(
    ChannelData->HostSessionID,
    0,
    Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostMouseButton,
    Payload);

  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostKeyboard:{

  auto Request = (Protocol_C2S_t::Channel_ScreenShare_View_ApplyToHostKeyboard_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  if(!(ChannelData->Flag & ProtocolChannel::ScreenShare::ChannelFlag::InputControl)){
    goto StateDone_gt;
  }

  Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostKeyboard_t Payload;
  Payload.ChannelID = ChannelID;
  Payload.Scancode = Request->Scancode;
  Payload.State = Request->State;

  Session::WriteCommand(
    ChannelData->HostSessionID,
    0,
    Protocol_S2C_t::Channel_ScreenShare_Share_ApplyToHostKeyboard,
    Payload);

  goto StateDone_gt;
}
case Protocol_C2S_t::RequestChannelList: {
  uint16_t channel_count = 0;
  
  auto nr = g_pile->ChannelList.GetNodeFirst();
  while (nr != g_pile->ChannelList.dst) {
    auto n = g_pile->ChannelList.GetNodeByReference(nr);
    channel_count++;
    nr = n->NextNodeReference;
  }
  
  size_t response_size = sizeof(Protocol_S2C_t::ChannelList_t) + 
                        (channel_count * sizeof(Protocol_S2C_t::ChannelInfo_t));
  
  uint8_t* response_buffer = (uint8_t*)malloc(response_size);
  if (!response_buffer) {
    goto StateDone_gt;
  }
  
  auto channel_list = (Protocol_S2C_t::ChannelList_t*)response_buffer;
  channel_list->ChannelCount = channel_count;
  
  auto channel_info_array = (Protocol_S2C_t::ChannelInfo_t*)(response_buffer + sizeof(Protocol_S2C_t::ChannelList_t));
  uint16_t info_index = 0;
  
  nr = g_pile->ChannelList.GetNodeFirst();
  while (nr != g_pile->ChannelList.dst && info_index < channel_count) {
    auto n = g_pile->ChannelList.GetNodeByReference(nr);
    
    Protocol_ChannelID_t channel_id(nr);
    auto& info = channel_info_array[info_index++];
      
    info.ChannelID = channel_id;
    info.Type = n->data.Type;
    info.UserCount = GetChannelUserCount(channel_id);
    info.HostSessionID = GetChannelHost(channel_id);
    info.IsPasswordProtected = 0;
    //
    const char* username = GetSessionUsername(GetChannelHost(channel_id));
    strncpy(info.Name, username, sizeof(info.Name) - 1);
    info.Name[sizeof(info.Name) - 1] = '\0';
    
    nr = n->NextNodeReference;
  }
  
  TCP_WriteCommand(
    peer,
    BasePacket->ID,
    Protocol_S2C_t::ChannelList,
    *channel_list
  );
  
  if (channel_count > 0) {
    TCP_write_DynamicPointer(peer, channel_info_array, channel_count * sizeof(Protocol_S2C_t::ChannelInfo_t));
  }
  
  free(response_buffer);
  
  goto StateDone_gt;
}

case Protocol_C2S_t::RequestChannelSessionList: {
  auto Request = (Protocol_C2S_t::RequestChannelSessionList_t*)RestPacket;
  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  
  if (IsChannelInvalid(ChannelID)) {
    // Send empty list for invalid channels
    Protocol_S2C_t::ChannelSessionList_t empty_list;
    empty_list.ChannelID = ChannelID;
    empty_list.SessionCount = 0;
    
    TCP_WriteCommand(
      peer,
      BasePacket->ID,
      Protocol_S2C_t::ChannelSessionList,
      empty_list
    );
    
    #if set_Verbose
      //_print("[RequestChannelSessionList] Invalid channel %d requested by session %d\n", (int)ChannelID, (int)SessionID);
    #endif
    
    goto StateDone_gt;
  }
  
  auto& Channel = g_pile->ChannelList[ChannelID];
  
  uint16_t session_count = 0;
  auto nr = Channel.SessionList.GetNodeFirst();
  while (nr != Channel.SessionList.dst) {
    auto n = Channel.SessionList.GetNodeByReference(nr);
    session_count++;
    nr = n->NextNodeReference;
  }
  
  size_t response_size = sizeof(Protocol_S2C_t::ChannelSessionList_t) + 
                        (session_count * sizeof(Protocol_S2C_t::SessionInfo_t));
  
  uint8_t* response_buffer = (uint8_t*)malloc(response_size);
  if (!response_buffer) {
    goto StateDone_gt;
  }
  
  auto session_list = (Protocol_S2C_t::ChannelSessionList_t*)response_buffer;
  session_list->ChannelID = ChannelID;
  session_list->SessionCount = session_count;
  
  auto session_info_array = (Protocol_S2C_t::SessionInfo_t*)(response_buffer + sizeof(Protocol_S2C_t::ChannelSessionList_t));
  uint16_t info_index = 0;
  
  Protocol_SessionID_t host_session = GetChannelHost(ChannelID);
  
  nr = Channel.SessionList.GetNodeFirst();
  while (nr != Channel.SessionList.dst && info_index < session_count) {
    auto n = Channel.SessionList.GetNodeByReference(nr);
    
    auto& info = session_info_array[info_index++];
    Protocol_ChannelSessionID_t channel_session_id(nr);
      
    info.SessionID = n->data.SessionID;
    info.ChannelSessionID = channel_session_id;
      
    auto& user_session = g_pile->SessionList[n->data.SessionID];
    info.AccountID = user_session.AccountID;
      
    info.IsHost = (n->data.SessionID == host_session) ? 1 : 0;
    info.JoinedAt = 0;
      
    const char* username = GetSessionUsername(n->data.SessionID);
    strncpy((char*)info.Username, username, sizeof(info.Username) - 1);
    info.Username[sizeof(info.Username) - 1] = '\0';
    
    nr = n->NextNodeReference;
  }
  
  TCP_WriteCommand(
    peer,
    BasePacket->ID,
    Protocol_S2C_t::ChannelSessionList,
    *session_list
  );
  
  if (session_count > 0) {
    TCP_write_DynamicPointer(peer, session_info_array, session_count * sizeof(Protocol_S2C_t::SessionInfo_t));
  }
  
  free(response_buffer);
  
  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_ViewToShare:{
  auto Request = (Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  
  // Send the flag from viewer to host
  Protocol_S2C_t::Channel_ScreenShare_ViewToShare_t Payload;
  Payload.ChannelID = ChannelID;
  Payload.Flag = Request->Flag;

  Session::WriteCommand(
    ChannelData->HostSessionID,
    0,
    Protocol_S2C_t::Channel_ScreenShare_ViewToShare,
    Payload);

  goto StateDone_gt;
}
case Protocol_C2S_t::Channel_ScreenShare_ShareToView:{
  auto Request = (Protocol_C2S_t::Channel_ScreenShare_ShareToView_t *)RestPacket;

  Protocol_ChannelID_t ChannelID = Request->ChannelID;
  if(IsChannelInvalid(ChannelID) == true){
    goto StateDone_gt;
  }
  auto Channel = &g_pile->ChannelList[ChannelID];
  auto ChannelData = (Channel_ScreenShare_Data_t *)Channel->Buffer;
  
  if(ChannelData->HostSessionID != SessionID){
    goto StateDone_gt;
  }

  auto nr = Channel->SessionList.GetNodeFirst();
  ChannelSessionList_Node_t *n;
  for(; nr != Channel->SessionList.dst; nr = n->NextNodeReference){
    n = Channel->SessionList.GetNodeByReference(nr);
    if(n->data.SessionID == SessionID){
      /* don't send to self (host) */
      continue;
    }

    Protocol_S2C_t::Channel_ScreenShare_ShareToView_t Payload;
    Payload.ChannelID = ChannelID;
    Payload.Flag = Request->Flag;

    Session::WriteCommand(
      n->data.SessionID,
      0,
      Protocol_S2C_t::Channel_ScreenShare_ShareToView,
      Payload);
  }

  goto StateDone_gt;
}
