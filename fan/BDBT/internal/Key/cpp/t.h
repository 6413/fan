gt_begin:
auto tp = &ta[Current];
while(
  LowHigh == 1 ?
  tp->k != (KeyNodeIterator_t)-1 :
  tp->k < _BDBT_set_ElementPerNode
){
  _BDBT_BP(Node_t) *Node = _BDBT_BP(GetNodeByReference)(list, tp->n);

  KeyNodeIterator_t tk = tp->k;
  LowHigh == 1 ? --tp->k : ++tp->k;

  _BDBT_BP(NodeReference_t) nnr = Node->n[tk];
  if(_BDBT_BP(inric)(list, nnr) == true){
    continue;
  }

  KeySize_t d8;
  KeySize_t m8;
  if constexpr(BitOrderMatters == true){
    d8 = (KeySize / 8 - 1) - (Current * BDBT_set_BitPerNode / 8);
    m8 = (8 - BDBT_set_BitPerNode) - Current * BDBT_set_BitPerNode % 8;
  }
  else{
    d8 = Current * BDBT_set_BitPerNode / 8;
    m8 = Current * BDBT_set_BitPerNode % 8;
  }

  ((uint8_t *)Key)[d8] ^= ((uint8_t *)Key)[d8] & _BDBT_set_ElementPerNode - 1 << m8;
  ((uint8_t *)Key)[d8] |= tk << m8;

  if(Current == KeySize / BDBT_set_BitPerNode - 1){
    Output = nnr;
    return 1;
  }

  tp = &ta[++Current];
  tp->n = nnr;
  tp->k = LowHigh == 1 ? _BDBT_set_ElementPerNode - 1 : 0;
}
if(Current == 0){
  return 0;
}
--Current;
goto gt_begin;
