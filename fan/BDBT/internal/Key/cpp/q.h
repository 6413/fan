#ifndef ENDIAN
  #error ENDIAN needs to be defined
#endif

#if defined(__compiler_clang) || 1
  uint8_t *kp8;
  if constexpr(BitOrderMatters == true && ENDIAN == 1){
    kp8 = &((uint8_t *)Key)[KeySize / 8 - 1];
  }
  else{
    kp8 = (uint8_t *)Key;
  }
  *KeyIndex = 0;
  while(*KeyIndex != KeySize){
    uint8_t Byte = *kp8;
    if constexpr(BitOrderMatters == true){
      Byte = ReverseKeyByte(Byte);
    }
    for(uint8_t i = 0; i < 8; i += BDBT_set_BitPerNode){
      uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
      _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, *cnr);
      _BDBT_BP(NodeReference_t) nnr = Node->n[k];
      if(_BDBT_BP(inric)(list, nnr) == true){
        *KeyIndex += i;
        return;
      }
      *cnr = nnr;
      Byte >>= BDBT_set_BitPerNode;
    }
    *KeyIndex += 8;
    if constexpr(BitOrderMatters == true && ENDIAN == 1){
      kp8--;
    }
    else{
      kp8++;
    }
  }
#elif defined(__compiler_gcc)
  /* this is faster with gcc, but needs assembly */
  #error this implement needs to care BitOrderMatters
  KeySize_t _KeyIndex = 0;
  while(_KeyIndex != KeySize / 8){
    uint8_t Byte = ((uint8_t *)Key)[_KeyIndex];
    for(uint8_t i = 0; i < 8; i += BDBT_set_BitPerNode){
      uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
      _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, *cnr);
      _BDBT_BP(NodeReference_t) nnr = Node->n[k];
      if(nnr == _BDBT_BP(GetNotValidNodeReference)(list)){
        *KeyIndex = _KeyIndex * 8 + i;
        return;
      }
      *cnr = nnr;
      Byte >>= BDBT_set_BitPerNode;
    }
    _KeyIndex++;
  }
  *KeyIndex = _KeyIndex * 8;
#endif
