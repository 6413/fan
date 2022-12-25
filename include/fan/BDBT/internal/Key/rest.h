static
void
_BDBT_P(KeyInFrom)
(
  _BDBT_BP(t) *list,
  void *Key,
  _BDBT_P(KeySize_t) KeyIndex,
  _BDBT_Key_ParameterKeySize(,__ca__)
  _BDBT_BP(NodeReference_t) cnr,
  _BDBT_BP(NodeReference_t) Output
){
  _BDBT_Key_PrepareBeforeLast
  uint8_t *kp8 = (uint8_t *)Key;
  kp8 += KeyIndex / 8;
  if(_BDBT_Key_GetBeforeLast != 0){
    if(KeyIndex < _BDBT_Key_GetBeforeLast){
      uint8_t m = KeyIndex % 8;
      uint8_t Byte = *kp8;
      Byte >>= m;
      for(uint8_t i = m; i < 8; i += BDBT_set_BitPerNode){
        uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
        _BDBT_BP(NodeReference_t) pnr = cnr;
        cnr = _BDBT_BP(NewNode)(list);
        _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, pnr);
        Node->n[k] = cnr;
        Byte >>= BDBT_set_BitPerNode;
      }
      KeyIndex += 8 - m;
      kp8++;
    }
  }
  if(_BDBT_Key_GetBeforeLast > 8){
    while(KeyIndex != _BDBT_Key_GetBeforeLast){
      uint8_t Byte = *kp8;
      for(uint8_t i = 0; i < 8; i += BDBT_set_BitPerNode){
        uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
        _BDBT_BP(NodeReference_t) pnr = cnr;
        cnr = _BDBT_BP(NewNode)(list);
        _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, pnr);
        Node->n[k] = cnr;
        Byte >>= BDBT_set_BitPerNode;
      }
      KeyIndex += 8;
      kp8++;
    }
  }
  {
    uint8_t Byte = *kp8;
    if(_BDBT_Key_GetBeforeLast == 0){
      Byte >>= KeyIndex;
      for(uint8_t i = KeyIndex; i < 8 - BDBT_set_BitPerNode; i += BDBT_set_BitPerNode){
        uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
        _BDBT_BP(NodeReference_t) pnr = cnr;
        cnr = _BDBT_BP(NewNode)(list);
        _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, pnr);
        Node->n[k] = cnr;
        Byte >>= BDBT_set_BitPerNode;
      }
    }
    else{
      uint8_t m = KeyIndex % 8;
      Byte >>= m;
      for(uint8_t i = m; i < 8 - BDBT_set_BitPerNode; i += BDBT_set_BitPerNode){
        uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
        _BDBT_BP(NodeReference_t) pnr = cnr;
        cnr = _BDBT_BP(NewNode)(list);
        _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, pnr);
        Node->n[k] = cnr;
        Byte >>= BDBT_set_BitPerNode;
      }
    }
    uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
    _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, cnr);
    Node->n[k] = Output;
  }
}

static
void
_BDBT_P(KeyQuery)
(
  _BDBT_BP(t) *list,
  void *Key,
  _BDBT_P(KeySize_t) *KeyIndex,
  _BDBT_Key_ParameterKeySize(,__ca__)
  _BDBT_BP(NodeReference_t) *cnr
){
  _BDBT_Key_PrepareBeforeLast
  uint8_t *kp8 = (uint8_t *)Key;
  *KeyIndex = 0;
  while(*KeyIndex != _BDBT_Key_GetKeySize){
    uint8_t Byte = *kp8;
    for(uint8_t i = 0; i < 8; i += BDBT_set_BitPerNode){
      uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
      _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, *cnr);
      _BDBT_BP(NodeReference_t) nnr = Node->n[k];
      if(nnr == _BDBT_BP(GetNotValidNodeReference)(list)){
        *KeyIndex += i;
        return;
      }
      *cnr = nnr;
      Byte >>= BDBT_set_BitPerNode;
    }
    *KeyIndex += 8;
    kp8++;
  }
}

static
void
_BDBT_P(KeyIn)
(
  _BDBT_BP(t) *list,
  void *Key,
  _BDBT_Key_ParameterKeySize(,__ca__)
  _BDBT_BP(NodeReference_t) cnr,
  _BDBT_BP(NodeReference_t) Output
){
  _BDBT_P(KeySize_t) KeyIndex;
  _BDBT_BP(NodeReference_t) nr = cnr;
  _BDBT_P(KeyQuery)(list, Key _BDBT_Key_PassKeySize, &KeyIndex, &nr);
  _BDBT_P(KeyInFrom)(list, Key _BDBT_Key_PassKeySize, KeyIndex, nr, Output);
}

static
void
_BDBT_P(KeyRemove)
(
  _BDBT_BP(t) *list,
  void *Key,
  _BDBT_Key_ParameterKeySize(,__ca__)
  _BDBT_BP(NodeReference_t) cnr
){
  uint8_t *kp8 = (uint8_t *)Key;

  _BDBT_BP(NodeReference_t) tna[_BDBT_Key_GetKeySize / BDBT_set_BitPerNode];
  uint8_t tka[_BDBT_Key_GetKeySize / BDBT_set_BitPerNode];

  _BDBT_P(KeySize_t) From = 0;
  {
    _BDBT_P(KeySize_t) KeyIndex = 0;
    while(KeyIndex != _BDBT_Key_GetKeySize / BDBT_set_BitPerNode){
      uint8_t Byte = *kp8;
      for(uint8_t i = 0; i < 8 / BDBT_set_BitPerNode; i++){
        uint8_t k = Byte & _BDBT_set_ElementPerNode - 1;
        tna[KeyIndex + i] = cnr;
        tka[KeyIndex + i] = k;
        _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, cnr);
        cnr = Node->n[k];
        for(_BDBT_BP(NodeEIT_t) ki = 0; ki < _BDBT_set_ElementPerNode; ki++){
          if(ki == k){
            continue;
          }
          if(Node->n[ki] != _BDBT_BP(GetNotValidNodeReference)(list)){
            From = KeyIndex + i;
            break;
          }
        }
        Byte >>= BDBT_set_BitPerNode;
      }
      KeyIndex += 8 / BDBT_set_BitPerNode;
      kp8++;
    }
  }

  {
    _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, tna[From]);
    Node->n[tka[From]] = _BDBT_BP(GetNotValidNodeReference)(list);
    ++From;
  }

  for(_BDBT_P(KeySize_t) i = From; i < _BDBT_Key_GetKeySize / BDBT_set_BitPerNode; i++){
    _BDBT_BP(Recycle)(list, tna[i]);
  }
}

#if BDBT_set_KeySize == 0
  typedef struct{
    _BDBT_BP(NodeReference_t) tna;
    uint8_t tka;
  }_BDBT_P(KeyTraverseA_t);

  static
  uintptr_t
  _BDBT_P(KeyTraverseANeededSpace)
  (
    _BDBT_P(KeySize_t) KeySize
  ){
    return KeySize / BDBT_set_BitPerNode;
  }
#endif

typedef struct{
  #if BDBT_set_KeySize != 0
    _BDBT_BP(NodeReference_t) tna[BDBT_set_KeySize / BDBT_set_BitPerNode];
    uint8_t tka[BDBT_set_KeySize / BDBT_set_BitPerNode];
  #endif
  _BDBT_P(KeySize_t) Current;

  _BDBT_BP(NodeReference_t) Output;
}_BDBT_P(KeyTraverse_t);

#if BDBT_set_KeySize == 0
  #define __a(i, n) KeyTraverseA[i].n
#else
  #define __a(i, n) KeyTraverse->n[i]
#endif

static
void
_BDBT_P(KeyTraverse_init)
(
  _BDBT_P(KeyTraverse_t) *KeyTraverse,
  #if BDBT_set_KeySize == 0
    _BDBT_P(KeyTraverseA_t) *KeyTraverseA,
  #endif
  _BDBT_BP(NodeReference_t) rnr
){
  KeyTraverse->Current = 0;
  __a(KeyTraverse->Current,tna) = rnr;
  __a(KeyTraverse->Current,tka) = 0;
}

static
bool
_BDBT_P(KeyTraverse)
(
  _BDBT_BP(t) *list,
  _BDBT_P(KeyTraverse_t) *KeyTraverse,
  #if BDBT_set_KeySize == 0
    _BDBT_P(KeyTraverseA_t) *KeyTraverseA,
  #endif
  void *Key
  _BDBT_Key_ParameterKeySize(__ca__,)
){
  gt_begin:
  while(__a(KeyTraverse->Current,tka) < _BDBT_set_ElementPerNode){
    _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, __a(KeyTraverse->Current,tna));

    uint8_t tk = __a(KeyTraverse->Current,tka)++;
    _BDBT_BP(NodeReference_t) nnr = Node->n[tk];
    if(nnr != _BDBT_BP(GetNotValidNodeReference)(list)){
      _BDBT_P(KeySize_t) d8 = KeyTraverse->Current * BDBT_set_BitPerNode / 8;
      _BDBT_P(KeySize_t) m8 = KeyTraverse->Current * BDBT_set_BitPerNode % 8;
      ((uint8_t *)Key)[d8] ^= ((uint8_t *)Key)[d8] & _BDBT_set_ElementPerNode - 1 << m8;
      ((uint8_t *)Key)[d8] |= tk << m8;
      if(KeyTraverse->Current == _BDBT_Key_GetKeySize / BDBT_set_BitPerNode - 1){
        KeyTraverse->Output = nnr;
        return 1;
      }
      KeyTraverse->Current++;
      __a(KeyTraverse->Current,tna) = nnr;
      __a(KeyTraverse->Current,tka) = 0;
    }
  }
  if(KeyTraverse->Current == 0){
    return 0;
  }
  --KeyTraverse->Current;
  goto gt_begin;
}

#undef __a
