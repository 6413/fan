template <uintptr_t KeySize>
struct _BDBT_P(Key_t){

  typedef std::conditional_t<
    KeySize <= 0xff,
    uint8_t,
      std::conditional_t<KeySize <= 0xffff,
      uint16_t,
      uint32_t
    >
  >KeySize_t;

  static constexpr KeySize_t BeforeLast = KeySize - 8;

  void
  InFrom
  (
    _BDBT_BP(t) *list,
    void *Key,
    KeySize_t KeyIndex,
    _BDBT_BP(NodeReference_t) cnr,
    _BDBT_BP(NodeReference_t) Output
  ){
    uint8_t *kp8 = (uint8_t *)Key;
    kp8 += KeyIndex / 8;
    if constexpr(BeforeLast != 0){
      if(KeyIndex < BeforeLast){
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
    if constexpr(BeforeLast > 8){
      while(KeyIndex != BeforeLast){
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
      if constexpr(BeforeLast == 0){
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

  void
  Query
  (
    _BDBT_BP(t) *list,
    void *Key,
    KeySize_t *KeyIndex,
    _BDBT_BP(NodeReference_t) *cnr
  ){
    uint8_t *kp8 = (uint8_t *)Key;
    *KeyIndex = 0;
    while(*KeyIndex != KeySize){
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

  void
  In
  (
    _BDBT_BP(t) *list,
    void *Key,
    _BDBT_BP(NodeReference_t) cnr,
    _BDBT_BP(NodeReference_t) Output
  ){
    KeySize_t KeyIndex;
    _BDBT_BP(NodeReference_t) nr = cnr;
    _BDBT_P(KeyQuery)(list, Key, &KeyIndex, &nr);
    _BDBT_P(KeyInFrom)(list, Key, KeyIndex, nr, Output);
  }

  void
  Remove
  (
    _BDBT_BP(t) *list,
    void *Key,
    _BDBT_BP(NodeReference_t) cnr
  ){
    uint8_t *kp8 = (uint8_t *)Key;

    _BDBT_BP(NodeReference_t) tna[KeySize / BDBT_set_BitPerNode];
    uint8_t tka[KeySize / BDBT_set_BitPerNode];

    KeySize_t From = 0;
    {
      KeySize_t KeyIndex = 0;
      while(KeyIndex != KeySize / BDBT_set_BitPerNode){
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

    for(KeySize_t i = From; i < KeySize / BDBT_set_BitPerNode; i++){
      _BDBT_BP(Recycle)(list, tna[i]);
    }
  }

  struct Traverse_t{
    _BDBT_BP(NodeReference_t) tna[KeySize / BDBT_set_BitPerNode];
    uint8_t tka[KeySize / BDBT_set_BitPerNode];
    KeySize_t Current;

    _BDBT_BP(NodeReference_t) Output;

    void
    init
    (
      _BDBT_BP(NodeReference_t) rnr
    ){
      this->Current = 0;
      this->tna[this->Current] = rnr;
      this->tka[this->Current] = 0;
    }

    bool
    Traverse
    (
      _BDBT_BP(t) *list,
      void *Key
    ){
      gt_begin:
      while(this->tka[this->Current] < _BDBT_set_ElementPerNode){
        _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, this->tna[this->Current]);

        uint8_t tk = this->tka[this->Current]++;
        _BDBT_BP(NodeReference_t) nnr = Node->n[tk];
        if(nnr != _BDBT_BP(GetNotValidNodeReference)(list)){
          KeySize_t d8 = this->Current * BDBT_set_BitPerNode / 8;
          KeySize_t m8 = this->Current * BDBT_set_BitPerNode % 8;
          ((uint8_t *)Key)[d8] ^= ((uint8_t *)Key)[d8] & _BDBT_set_ElementPerNode - 1 << m8;
          ((uint8_t *)Key)[d8] |= tk << m8;
          if(this->Current == KeySize / BDBT_set_BitPerNode - 1){
            this->Output = nnr;
            return 1;
          }
          this->Current++;
          this->tna[this->Current] = nnr;
          this->tka[this->Current] = 0;
        }
      }
      if(this->Current == 0){
        return 0;
      }
      --this->Current;
      goto gt_begin;
    }
  };
};
