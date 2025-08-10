namespace TCPPeer{

void Open(NET_TCP_peer_t *peer){
  auto SockData = (TCPMain_SockData_t *)NET_TCP_GetSockData(g_pile->TCP, g_pile->extid);
  auto PeerData = (TCPMain_PeerData_t *)NET_TCP_GetPeerData(peer, g_pile->extid);

  if(NET_TCP_MakePeerNoDelay(peer) != 0){
    __abort();
  }
  NET_TCP_StartReadLayer(peer, SockData->ReadLayerID);

  PeerData->state = PeerState_Idle_e;

  PeerData->SessionID = Session::OpenFromTCPPeer(peer);
}

void Close(NET_TCP_peer_t *peer){
  auto PeerData = (TCPMain_PeerData_t *)NET_TCP_GetPeerData(peer, g_pile->extid);

  Session::Close(PeerData->SessionID);

  switch(PeerData->state){
    case PeerState_Idle_e:{
      break;
    }
    case PeerState_Waitting_BasePacket_e:
    case PeerState_Waitting_Data_e:
    {
      A_resize(PeerData->Buffer, 0);
      break;
    }
  }
}

} // namespace TCPPeer
