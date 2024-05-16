#ifndef ENDIAN
  #error ENDIAN needs to be defined
#endif

uint8_t *kp8;
if constexpr(BitOrderMatters == true && ENDIAN == 1){
  kp8 = &((uint8_t *)Key)[KeySize / 8 - 1];
}
else{
  kp8 = (uint8_t *)Key;
}
KeySize_t KeyIndex = 0;
while(KeyIndex != KeySize){
  uint8_t Byte = *kp8;
  if constexpr(BitOrderMatters == true){
    Byte = ReverseKeyByte(Byte);
  }
  for(uint8_t i = 0; i < 8; i += BDBT_set_BitPerNode){
    uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
    _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, *cnr);
    *cnr = Node->n[k];
    Byte >>= BDBT_set_BitPerNode;
  }
  KeyIndex += 8;
  if constexpr(BitOrderMatters == true && ENDIAN == 1){
    kp8--;
  }
  else{
    kp8++;
  }
}
