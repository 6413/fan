#ifndef ENDIAN
  #error ENDIAN needs to be defined
#endif

uint8_t *kp8 = (uint8_t *)Key;
if constexpr(BitOrderMatters == true && ENDIAN == 1){
  kp8 = &((uint8_t *)Key)[KeySize / 8 - 1];
}
else{
  kp8 = (uint8_t *)Key;
}

_BDBT_BP(NodeReference_t) tna[KeySize / BDBT_set_BitPerNode];
uint8_t tka[KeySize / BDBT_set_BitPerNode];

KeySize_t From = 0;
{
  KeySize_t KeyIndex = 0;
  while(KeyIndex != KeySize / BDBT_set_BitPerNode){
    uint8_t Byte = *kp8;
    if constexpr(BitOrderMatters == true){
      Byte = ReverseKeyByte(Byte);
    }
    for(uint8_t i = 0; i < 8 / BDBT_set_BitPerNode; i++){
      uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
      tna[KeyIndex + i] = *cnr;
      tka[KeyIndex + i] = k;
      _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, *cnr);
      *cnr = Node->n[k];
      for(_BDBT_BP(NodeEIT_t) ki = 0; ki < _BDBT_set_ElementPerNode; ki++){
        if(ki == k){
          continue;
        }
        if(_BDBT_BP(inric)(list, Node->n[ki]) == false){
          From = KeyIndex + i;
          break;
        }
      }
      Byte >>= BDBT_set_BitPerNode;
    }
    KeyIndex += 8 / BDBT_set_BitPerNode;
    if constexpr(BitOrderMatters == true && ENDIAN == 1){
      kp8--;
    }
    else{
      kp8++;
    }
  }
}

{
  _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, tna[From]);
  Node->n[tka[From]] = _BDBT_BP(gnric)(list);
  ++From;
}

for(KeySize_t i = From; i < KeySize / BDBT_set_BitPerNode; i++){
  _BDBT_BP(Recycle)(list, tna[i]);
}

return From * BDBT_set_BitPerNode;
