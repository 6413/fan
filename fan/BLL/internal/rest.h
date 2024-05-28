/* too much overenginer here */

#if BLL_set_StoreFormat == 1
  /* TODO can be more smaller */
  typedef BLL_set_type_node _P(BlockIndex_t);

  #if BLL_set_StoreFormat1_ElementPerBlock <= 0xff
    typedef uint8_t _P(BlockModulo_t);
  #elif BLL_set_StoreFormat1_ElementPerBlock <= 0xffff
    typedef uint16_t _P(BlockModulo_t);
  #elif BLL_set_StoreFormat1_ElementPerBlock <= 0xffffffff
    typedef uint32_t _P(BlockModulo_t);
  #else
    #error no
  #endif
#endif

#if BLL_set_PadNode == 0
  #pragma pack(push, 1)
#endif
#if defined(_BLL_HaveConstantNodeData)
  #if defined(BLL_set_NodeData)
    BLL_StructBegin(_P(NodeData_t))
      BLL_set_NodeData
    BLL_StructEnd(_P(NodeData_t))
  #elif defined(BLL_set_NodeDataType)
    typedef BLL_set_NodeDataType _P(NodeData_t);
  #else
    #error not possible
  #endif
#else
  typedef void _P(NodeData_t);
#endif
BLL_StructBegin(_P(Node_t))
  #if BLL_set_Link == 1
    #if BLL_set_PreferNextFirst == 1
      _P(NodeReference_t) NextNodeReference;
      _P(NodeReference_t) PrevNodeReference;
    #else
      _P(NodeReference_t) PrevNodeReference;
      _P(NodeReference_t) NextNodeReference;
    #endif
  #endif
  #ifdef _BLL_HaveConstantNodeData
    #if BLL_set_CPP_Node_ConstructDestruct
      _P(NodeData_t) data;
      #if BLL_set_Link == 0
        uint8_t _PaddingForNextRecycled[
          sizeof(_P(NodeData_t)) < sizeof(_P(NodeReference_t)) ?
            sizeof(_P(NodeReference_t)) - sizeof(_P(NodeData_t)) :
            0
        ];
      #endif
    #else
      union{
        _P(NodeData_t) data;
        #if BLL_set_Link == 0
          /* used for empty next */
          _P(NodeReference_t) _NextRecycled;
        #endif
      };
    #endif
  #endif
BLL_StructEnd(_P(Node_t))
#if BLL_set_PadNode == 0
  #pragma pack(pop)
#endif

#if defined(BLL_set_MultipleType_Sizes)
  static constexpr auto _P(MultipleType_MakeArray)(auto... x) {
    std::array<std::tuple_element_t<0, std::tuple<decltype(x)...>>, sizeof...(x)> arr = {
      x...
    };
    #if BLL_set_Link == 1
    arr[BLL_set_MultipleType_LinkIndex] += sizeof(_P(NodeReference_t)) * 2;
    /*
    uintptr_t NeededSize = sizeof(_P(NodeReference_t)) * 2;
    bool found = false;
    for(uintptr_t i = 0; i < arr.size(); i++){
      if(arr[i] >= NeededSize){
        found = true;
      }
    }
    if(found == false){
      uintptr_t BestDiff = -1;
      uintptr_t BestIndex;
      for(uintptr_t i = 0; i < arr.size(); i++){
        if(NeededSize - arr[i])
      }
    }
    */
    #endif
    return arr;
  }
  static constexpr auto _P(_MultipleType_Sizes) = _P(MultipleType_MakeArray)(BLL_set_MultipleType_Sizes);
#endif

#if BLL_set_StoreFormat == 0
  #define BVEC_set_prefix _P(_NodeList)
  #define BVEC_set_NodeType BLL_set_type_node
  #if defined(BLL_set_MultipleType_Sizes)
    #define BVEC_set_MultipleType
    #define BVEC_set_MultipleType_SizeArray _P(_MultipleType_Sizes)
  #elif defined(_BLL_HaveConstantNodeData)
    #define BVEC_set_NodeData _P(Node_t)
  #endif
  #if BLL_set_CPP_CopyAtPointerChange
    #define BVEC_set_HandleAllocate 0
  #endif
  #define BVEC_set_alloc_open BLL_set_alloc_open
  #define BVEC_set_alloc_resize BLL_set_alloc_resize
  #define BVEC_set_alloc_close BLL_set_alloc_close
  #include <BVEC/BVEC.h>
#elif BLL_set_StoreFormat == 1
  #ifndef _BLL_HaveConstantNodeData
    #error not yet
  #endif
  #if defined(BLL_set_MultipleType_Sizes)
    #error not yet
  #endif
  #define BVEC_set_prefix _P(_BlockList)
  #define BVEC_set_NodeType BLL_set_type_node
  #define BVEC_set_NodeData _P(Node_t) *
  #define BVEC_set_alloc_open BLL_set_alloc_open
  #define BVEC_set_alloc_close BLL_set_alloc_close
  #include <BVEC/BVEC.h>
#endif

#if defined(BLL_set_MultipleType_Sizes)
  consteval
  static
  uintptr_t
  _P(_MultipleType_GetLinkIndex)(
  ){
    #if defined(BLL_set_MultipleType_LinkIndex)
      return BLL_set_MultipleType_LinkIndex;
    #else
      uintptr_t BestSize = -1;
      uintptr_t BestIndex;
      for(
        uintptr_t i = 0;
        i < sizeof(_P(_MultipleType_Sizes)) / sizeof(_P(_MultipleType_Sizes)[0]);
        i++
      ){
        if(_P(_MultipleType_Sizes)[i] < BestSize){
          BestSize = _P(_MultipleType_Sizes)[i];
          BestIndex = i;
        }
      }
      return BestIndex;
    #endif
  }
#endif

BLL_StructBegin(_P(t))
  #if BLL_set_StoreFormat == 0
    _P(_NodeList_t) NodeList;
  #elif BLL_set_StoreFormat == 1
    _P(_BlockList_t) BlockList;
  #endif
  #if BLL_set_StoreFormat == 1
    BLL_set_type_node NodeCurrent;
  #endif
  #ifndef _BLL_HaveConstantNodeData
    uint32_t NodeSize;
  #endif
  #if BLL_set_LinkSentinel
    _P(NodeReference_t) src;
    _P(NodeReference_t) dst;
  #endif
  #if BLL_set_Recycle
    struct{
      _P(NodeReference_t) c;
      BLL_set_type_node p;
    }e;
  #endif
  #if BLL_set_SafeNext == 1
    _P(NodeReference_t) SafeNext;
  #elif BLL_set_SafeNext > 1
    uint8_t SafeNextCount;
    _P(NodeReference_t) SafeNext[BLL_set_SafeNext];
  #endif
#if BLL_set_Language == 0
  BLL_StructEnd(_P(t))
#else
  using nd_t = _P(NodeData_t);
  using nr_t = _P(NodeReference_t);

  nr_t gnric(){
    return _P(gnric)();
  }
  bool inric(
    nr_t nr
  ){
    return _P(inric)(nr);
  }
#endif

_BLL_fdecpi(BLL_set_NodeSizeType, GetNodeSize
){
  /* TODO */
  /* this function doesnt check store format */
  /* can give compile error in future */

  #if defined(BLL_set_MultipleType_Sizes)
    return _P(_NodeList_GetNodeSize)(&_BLL_this->NodeList, PointerIndex);
  #elif defined(_BLL_HaveConstantNodeData)
    return sizeof(_P(Node_t));
  #else
    return _P(_NodeList_GetNodeSize)(&_BLL_this->NodeList);
  #endif
}

/* is node reference invalid */
_BLL_fdec(bool, inri,
  _P(NodeReference_t) nr
){
  #if BLL_set_StoreFormat == 0
    return *_P(gnrint)(&nr) >= _BLL_this->NodeList.Current;
  #elif BLL_set_StoreFormat == 1
    return *_P(gnrint)(&nr) >= _BLL_this->NodeCurrent;
  #endif
}

_BLL_fdecpi(_P(Node_t) *, _GetNodeByReference,
  _P(NodeReference_t) nr
){
  #if BLL_set_StoreFormat == 0
    return (_P(Node_t) *)_P(_NodeList_GetNode)(
      &_BLL_this->NodeList,
      *_P(gnrint)(&nr)
      #if defined(BLL_set_MultipleType_Sizes)
        , PointerIndex
      #endif
    );
  #elif BLL_set_StoreFormat == 1
    #if defined(_BLL_HaveConstantNodeData)
      _P(BlockIndex_t) bi = *_P(gnrint)(&nr) / BLL_set_StoreFormat1_ElementPerBlock;
      _P(BlockModulo_t) bm = *_P(gnrint)(&nr) % BLL_set_StoreFormat1_ElementPerBlock;
      /* TODO this looks like mess check it */
      _P(Node_t) *n = &((_P(Node_t) *)((void **)&_BLL_this->BlockList.ptr[0])[bi])[bm];
      return n;
    #else
      #error not implemented yet
    #endif
  #endif
}

/* get node reference th of node reference */
_BLL_fdec(_P(NodeReference_t) *, _GetNRTHOfNR,
  _P(NodeReference_t) nr,
  BLL_set_type_node th
){
  #if defined(BLL_set_MultipleType_Sizes)
    _P(Node_t) *n = _BLL_fcall(_GetNodeByReference, _P(_MultipleType_GetLinkIndex)(), nr);
  #else
    _P(Node_t) *n = _BLL_fcall(_GetNodeByReference, nr);
  #endif
  return &((_P(NodeReference_t) *)n)[th];
}

_BLL_fdecpi(_P(Node_t) *, GetNodeByReference,
  _P(NodeReference_t) nr
){
  return _BLL_fcallpi(_GetNodeByReference, nr);
}

_BLL_fdecpi(_P(NodeData_t) *, GetNodeReferenceData,
  _P(NodeReference_t) NodeReference
){
  _P(Node_t) *Node = _BLL_fcallpi(GetNodeByReference, NodeReference);
  #if defined(BLL_set_MultipleType_Sizes)
    #if BLL_set_Link == 1
      if(PointerIndex == _P(_MultipleType_GetLinkIndex)()){
        return (_P(NodeData_t) *)((uint8_t *)Node + sizeof(_P(Node_t)));
      }
      else{
        return (_P(NodeData_t) *)Node;
      }
    #else
      return (_P(NodeData_t) *)Node;
    #endif
  #elif defined(_BLL_HaveConstantNodeData)
    return (_P(NodeData_t) *)&Node->data;
  #else
    return (_P(NodeData_t) *)(
      (uint8_t *)Node
      #if BLL_set_Link == 1
        + sizeof(_P(Node_t))
      #endif
    );
  #endif
}

_BLL_fdec(BLL_set_type_node, Usage
){
  return
  #if BLL_set_StoreFormat == 0
    _BLL_this->NodeList.Current
  #elif BLL_set_StoreFormat == 1
    _BLL_this->NodeCurrent
  #endif
  #if BLL_set_Recycle
    - _BLL_this->e.p
  #endif
  #if BLL_set_LinkSentinel
    - 2
  #endif
  ;
}

#if BLL_set_StoreFormat == 1
  _BLL_fdec(void, _PushNewBlock
  ){
    _P(Node_t) *n = (_P(Node_t) *)BLL_set_alloc_open(
      sizeof(_P(Node_t)) * BLL_set_StoreFormat1_ElementPerBlock
    );
    _P(_BlockList_Add)(&_BLL_this->BlockList, &n);
  }
#endif

_BLL_fdec(void, _Node_Construct,
  _P(NodeReference_t) nr
){
  #if BLL_set_CPP_Node_ConstructDestruct
    #if defined(BLL_set_MultipleType_LinkIndex)
      #error implement this
    #else
      new (_BLL_fcall(GetNodeReferenceData, nr)) _P(NodeData_t);
    #endif
  #endif
}
_BLL_fdec(void, _Node_Destruct,
  _P(NodeReference_t) nr
){
  #if BLL_set_CPP_Node_ConstructDestruct
    #if defined(BLL_set_MultipleType_LinkIndex)
      #error implement this
    #else
  printf("- %p %d\n", this, nr);
      _BLL_fcall(GetNodeReferenceData, nr)->~_P(NodeData_t)();
    #endif
  #endif
}

#if BLL_set_Recycle
  /* get recycle node reference of node reference */
  _BLL_fdec(_P(NodeReference_t) *, _grecnrofnr,
    _P(NodeReference_t) nr
  ){
    return _BLL_fcall(_GetNRTHOfNR, nr, 0);
  }

  #if BLL_set_IsNodeRecycled == 1
    /* get recycle value */
    _BLL_fdec(_P(NodeReference_t), _grv,
    ){
      return _P(gnric)();
    }
    /* get non recycle value */
    _BLL_fdec(_P(NodeReference_t), _gnrv,
    ){
      _P(NodeReference_t) nr;
      *_P(gnrint)(&nr) = (BLL_set_type_node)-2;
      return nr;
    }

    /* get recycle value of node reference */
    _BLL_fdec(_P(NodeReference_t) *, _grvonr,
      _P(NodeReference_t) *nr
    ){
      return _BLL_fcall(_GetNRTHOfNR, *nr, 1);
    }

    _BLL_fdec(bool, IsNodeReferenceRecycled,
      _P(NodeReference_t) nr
    ){
      return _P(inre)(nr, _BLL_fcall(_grv));
    }
  #endif

  _BLL_fdec(void, _Recycle,
    _P(NodeReference_t) nr
  ){
    _P(NodeReference_t) *NextRecycled = _BLL_fcall(_grecnrofnr, nr);

    *NextRecycled = _BLL_this->e.c;
    #if BLL_set_IsNodeRecycled == 1
      *_BLL_fcall(_grvonr, &nr) = _BLL_fcall(_grv);
    #endif
    _BLL_this->e.c = nr;
    _BLL_this->e.p++;
  }

  _BLL_fdec(void, Recycle,
    _P(NodeReference_t) nr
  ){
    _BLL_fcall(_Node_Destruct, nr);
    _BLL_fcall(_Recycle, nr);
  }

  _BLL_fdec(void, Recycle_NoDestruct,
    _P(NodeReference_t) nr
  ){
    _BLL_fcall(_Recycle, nr);
  }
#endif

#if BLL_set_Recycle
  _BLL_fdec(_P(NodeReference_t), _NewNode_empty_NoConstruct
  ){
    _P(NodeReference_t) nr = _BLL_this->e.c;
    _BLL_this->e.c = *_BLL_fcall(_grecnrofnr, nr);
    _BLL_this->e.p--;
    return nr;
  }
#endif
_BLL_fdec(_P(NodeReference_t), _NewNode_alloc_NoConstruct
){
  _P(NodeReference_t) r;
  #if BLL_set_StoreFormat == 0
    *_P(gnrint)(&r) = _BLL_this->NodeList.Current;

    #if BLL_set_CPP_CopyAtPointerChange
      if(NodeList.Current == NodeList.Possible){
        AllocateNewBuffer(NodeList.Current + 1);
      }
    #endif

    _P(_NodeList_AddEmpty)(&_BLL_this->NodeList, 1);

  #elif BLL_set_StoreFormat == 1
    if(_BLL_this->NodeCurrent % BLL_set_StoreFormat1_ElementPerBlock == 0){
      _BLL_fcall(_PushNewBlock);
    }
    *_P(gnrint)(&r) = _BLL_this->NodeCurrent++;
  #endif
  return r;
}
_BLL_fdec(_P(NodeReference_t), _NewNode_NoConstruct
){
  #if BLL_set_Recycle
    if(_BLL_this->e.p){
      return _BLL_fcall(_NewNode_empty_NoConstruct);
    }
  #endif
  return _BLL_fcall(_NewNode_alloc_NoConstruct);
}
_BLL_fdec(_P(NodeReference_t), NewNode
){
  _P(NodeReference_t) nr = _BLL_fcall(_NewNode_NoConstruct);
  #if BLL_set_IsNodeRecycled == 1
    *_BLL_fcall(_grvonr, &nr) = _BLL_fcall(_gnrv);
  #endif
  _BLL_fcall(_Node_Construct, nr);
  printf("+%p %d\n", this, nr);
  return nr;
}
_BLL_fdec(void, NewTillUsage,
  BLL_set_type_node Amount
){
  while(_BLL_fcall(Usage) < Amount){
    _BLL_fcall(NewNode);
  }
}

#if BLL_set_Link == 1
  /* TODO implement _gln better */
  /* get linked node */
  _BLL_fdec(_P(Node_t) *, _gln,
    _P(NodeReference_t) nr
  ){
    #if defined(BLL_set_MultipleType_Sizes)
      return _BLL_fcall(_GetNodeByReference, _P(_MultipleType_GetLinkIndex)(), nr);
    #else
      return _BLL_fcall(_GetNodeByReference, nr);
    #endif
  }
  /* get linked node */
  _BLL_fdec(_P(Node_t) *, gln,
    _P(NodeReference_t) nr
  ){
    #if defined(BLL_set_MultipleType_Sizes)
      return _BLL_fcall(GetNodeByReference, _P(_MultipleType_GetLinkIndex)(), nr);
    #else
      return _BLL_fcall(GetNodeByReference, nr);
    #endif
  }

  _BLL_fdec(void, linkNext,
    _P(NodeReference_t) srcnr,
    _P(NodeReference_t) dstnr
  ){
    _P(Node_t) *srcNode = _BLL_fcall(gln, srcnr);
    _P(Node_t) *dstNode = _BLL_fcall(_gln, dstnr);
    _P(NodeReference_t) nnr = srcNode->NextNodeReference;
    _P(Node_t) *nextNode = _BLL_fcall(gln, nnr);
    srcNode->NextNodeReference = dstnr;
    dstNode->PrevNodeReference = srcnr;
    dstNode->NextNodeReference = nnr;
    nextNode->PrevNodeReference = dstnr;
  }
  _BLL_fdec(void, linkNextOfOrphan,
    _P(NodeReference_t) srcnr,
    _P(NodeReference_t) dstnr
  ){
    _P(Node_t) *srcNode = _BLL_fcall(gln, srcnr);
    _P(Node_t) *dstNode = _BLL_fcall(_gln, dstnr);
    srcNode->NextNodeReference = dstnr;
    dstNode->PrevNodeReference = srcnr;
  }
  _BLL_fdec(void, linkPrev,
    _P(NodeReference_t) srcnr,
    _P(NodeReference_t) dstnr
  ){
    _P(Node_t) *srcNode = _BLL_fcall(gln, srcnr);
    _P(NodeReference_t) pnr = srcNode->PrevNodeReference;
    _P(Node_t) *prevNode = _BLL_fcall(gln, pnr);
    prevNode->NextNodeReference = dstnr;
    _P(Node_t) *dstNode = _BLL_fcall(_gln, dstnr);
    dstNode->PrevNodeReference = pnr;
    dstNode->NextNodeReference = srcnr;
    srcNode->PrevNodeReference = dstnr;
  }
  _BLL_fdec(void, linkPrevOfOrphan,
    _P(NodeReference_t) srcnr,
    _P(NodeReference_t) dstnr
  ){
    _P(Node_t) *srcNode = _BLL_fcall(gln, srcnr);
    _P(Node_t) *dstNode = _BLL_fcall(_gln, dstnr);
    dstNode->NextNodeReference = srcnr;
    srcNode->PrevNodeReference = dstnr;
  }

  _BLL_fdec(void, Unlink,
    _P(NodeReference_t) nr
  ){
    _P(Node_t) *Node = _BLL_fcall(gln, nr);

    #if BLL_set_SafeNext == 1
      if(_BLL_this->SafeNext == nr){
        _BLL_this->SafeNext = Node->PrevNodeReference;
      }
    #elif BLL_set_SafeNext > 1
      for(uint8_t i = 0; i < _BLL_this->SafeNextCount; i++){
        if(_BLL_this->SafeNext[i] == nr){
          _BLL_this->SafeNext[i] = Node->PrevNodeReference;
        }
      }
    #endif

    _P(NodeReference_t) nnr = Node->NextNodeReference;
    _P(NodeReference_t) pnr = Node->PrevNodeReference;
    _BLL_fcall(gln, pnr)->NextNodeReference = nnr;
    _BLL_fcall(gln, nnr)->PrevNodeReference = pnr;
  }

  /* unlink recycle */
  _BLL_fdec(void, unlrec,
    _P(NodeReference_t) nr
  ){
    _BLL_fcall(Unlink, nr);
    _BLL_fcall(Recycle, nr);
  }

  #if BLL_set_LinkSentinel
    _BLL_fdec(_P(NodeReference_t), GetNodeFirst
    ){
      return _BLL_fcall(_gln, _BLL_this->src)->NextNodeReference;
    }
    _BLL_fdec(_P(NodeReference_t), GetNodeLast
    ){
      return _BLL_fcall(_gln, _BLL_this->dst)->PrevNodeReference;
    }

    _BLL_fdec(_P(NodeReference_t), NewNodeFirst_empty
    ){
      _P(NodeReference_t) nr = _BLL_fcall(_NewNode_empty_NoConstruct);
      _P(NodeReference_t) srcnr = _BLL_this->src;
      _BLL_fcall(_gln, nr)->NextNodeReference = srcnr;
      _BLL_fcall(_gln, srcnr)->PrevNodeReference = nr;
      _BLL_this->src = nr;
      _BLL_fcall(_Node_Construct, srcnr);
      return srcnr;
    }
    _BLL_fdec(_P(NodeReference_t), NewNodeFirst_alloc
    ){
      _P(NodeReference_t) nr = _BLL_fcall(_NewNode_alloc_NoConstruct);
      _P(NodeReference_t) srcnr = _BLL_this->src;
      _BLL_fcall(_gln, nr)->NextNodeReference = srcnr;
      _BLL_fcall(_gln, srcnr)->PrevNodeReference = nr;
      _BLL_this->src = nr;
      _BLL_fcall(_Node_Construct, srcnr);
      return srcnr;
    }
    _BLL_fdec(_P(NodeReference_t), NewNodeFirst
    ){
      if(_BLL_this->e.p){
        return _BLL_fcall(NewNodeFirst_empty);
      }
      else{
        return _BLL_fcall(NewNodeFirst_alloc);
      }
    }
    _BLL_fdec(_P(NodeReference_t), NewNodeLast_empty
    ){
      _P(NodeReference_t) nr = _BLL_fcall(_NewNode_empty_NoConstruct);
      _P(NodeReference_t) dstnr = _BLL_this->dst;
      _BLL_fcall(_gln, nr)->PrevNodeReference = dstnr;
      _BLL_fcall(_gln, dstnr)->NextNodeReference = nr;
      _BLL_this->dst = nr;
      _BLL_fcall(_Node_Construct, dstnr);
      return dstnr;
    }
    _BLL_fdec(_P(NodeReference_t), NewNodeLast_alloc
    ){
      _P(NodeReference_t) nr = _BLL_fcall(_NewNode_alloc_NoConstruct);
      _P(NodeReference_t) dstnr = _BLL_this->dst;
      _BLL_fcall(_gln, nr)->PrevNodeReference = dstnr;
      _BLL_fcall(_gln, dstnr)->NextNodeReference = nr;
      _BLL_this->dst = nr;
      _BLL_fcall(_Node_Construct, dstnr);
      return dstnr;
    }
    _BLL_fdec(_P(NodeReference_t), NewNodeLast
    ){
      if(_BLL_this->e.p){
        return _BLL_fcall(NewNodeLast_empty);
      }
      else{
        return _BLL_fcall(NewNodeLast_alloc);
      }
    }

    _BLL_fdec(bool, IsNRSentinel,
      _P(NodeReference_t) nr
    ){
      if(*_P(gnrint)(&nr) == *_P(gnrint)(&_BLL_this->src)){
        return 1;
      }
      if(*_P(gnrint)(&nr) == *_P(gnrint)(&_BLL_this->src)){
        return 1;
      }
      return 0;
    }

    _BLL_fdec(void, LinkAsFirst,
      _P(NodeReference_t) nr
    ){
      _BLL_fcall(linkNext, _BLL_this->src, nr);
    }
    _BLL_fdec(void, LinkAsLast,
      _P(NodeReference_t) nr
    ){
      _BLL_fcall(linkPrev, _BLL_this->dst, nr);
    }

    _BLL_fdec(void, ReLinkAsFirst,
      _P(NodeReference_t) nr
    ){
      _BLL_fcall(Unlink, nr);
      _BLL_fcall(LinkAsFirst, nr);
    }
    _BLL_fdec(void, ReLinkAsLast,
      _P(NodeReference_t) nr
    ){
      _BLL_fcall(Unlink, nr);
      _BLL_fcall(LinkAsLast, nr);
    }
  #endif
#endif

#include "nrtra.h"

#if BLL_set_CPP_CopyAtPointerChange
  #if !defined(_BLL_HaveConstantNodeData)
    #error those who trigger this, will burn in hell
  #endif

  _BLL_fdec(void, AllocateNewBuffer,
    BLL_set_type_node Amount
  ){
    _P(_NodeList_SetPossibleWith)(&NodeList, Amount);
    void *np = BLL_set_alloc_open(NodeList.Possible * sizeof(_P(Node_t)));
    __MemoryCopy(NodeList.ptr, np, NodeList.Current * sizeof(_P(Node_t)));

    _P(nrtra_t) nrtra;
    nrtra.Open(_BLL_this);
    while(nrtra.Loop(_BLL_this) == true){
      new
        (&((_P(Node_t) *)np)[*_P(gnrint)(&nrtra.nr)].data)
        _P(NodeData_t)(((_P(Node_t) *)NodeList.ptr)[*_P(gnrint)(&nrtra.nr)].data);
    }
    nrtra.Close(_BLL_this);

    BLL_set_alloc_close(NodeList.ptr);
    _P(_NodeList_SetPointer)(&_BLL_this->NodeList, np);
  }
#endif

_BLL_fdec(void, _DestructAllNodes
){
  #if BLL_set_CPP_Node_ConstructDestruct
    _P(nrtra_t) nrtra;
    nrtra.Open(_BLL_this);
    while(nrtra.Loop(_BLL_this) == true){
      _BLL_fcall(_Node_Destruct, nrtra.nr);
    }
    nrtra.Close(_BLL_this);
  #endif
}

#if BLL_set_StoreFormat == 1
  _BLL_fdec(void, _StoreFormat1_CloseAllocatedBlocks
  ){
    _P(BlockIndex_t) BlockAmount = _BLL_this->BlockList.Current;
    for(_P(BlockIndex_t) i = 0; i < BlockAmount; i++){
      /* TODO looks ugly cast */
      void *p = (void *)((_P(Node_t) **)&_BLL_this->BlockList.ptr[0])[i];
      BLL_set_alloc_close(p);
    }
  }
#endif

_BLL_fdec(void, _AfterInitNodes
){
  #if BLL_set_Recycle
    _BLL_this->e.p = 0;
  #endif

  #if BLL_set_StoreFormat == 0
    #if BLL_set_LinkSentinel
      #if BLL_set_CPP_CopyAtPointerChange
        /* TOOD better do this at Open */
        if(NodeList.Possible < 2){
          _BLL_fcall(AllocateNewBuffer, 2);
        }
      #endif

      _P(_NodeList_AddEmpty)(&_BLL_this->NodeList, 2);

      *_P(gnrint)(&_BLL_this->src) = 0;
      *_P(gnrint)(&_BLL_this->dst) = 1;
    #endif
  #elif BLL_set_StoreFormat == 1
    #if BLL_set_LinkSentinel
      _BLL_this->src = _BLL_fcall(_NewNode_NoConstruct);
      _BLL_this->dst = _BLL_fcall(_NewNode_NoConstruct);
    #endif
  #endif

  #if BLL_set_LinkSentinel
    _BLL_fcall(_gln, _BLL_this->src)->NextNodeReference = _BLL_this->dst;
    _BLL_fcall(_gln, _BLL_this->dst)->PrevNodeReference = _BLL_this->src;
  #endif
}

#if BLL_set_CPP_ConstructDestruct
  private:
#endif
_BLL_fdecnds(void, Open
){
  BLL_set_NodeSizeType NodeSize = sizeof(_P(Node_t));
  #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
    NodeSize += NodeDataSize;
    #if BLL_set_Link == 0
      /* for NextRecycled */
      #if BLL_set_Recycle
        NodeSize += sizeof(_P(Node_t)) < sizeof(_P(NodeReference_t)) ? sizeof(_P(Node_t)) - sizeof(_P(NodeReference_t)) : 0;
      #endif
    #endif
  #endif

  #if __sanit
    *_P(gnrint)(&_BLL_this->e.c) = 0;
  #endif

  #if BLL_set_StoreFormat == 0
    #if defined(_BLL_HaveConstantNodeData) || defined(BLL_set_MultipleType_Sizes)
      _P(_NodeList_Open)(&_BLL_this->NodeList);
    #else
      _P(_NodeList_Open)(&_BLL_this->NodeList, NodeSize);
    #endif
  #elif BLL_set_StoreFormat == 1
    _BLL_this->NodeCurrent = 0;
    _P(_BlockList_Open)(&_BLL_this->BlockList);
  #endif
  _BLL_fcall(_AfterInitNodes);

  #if BLL_set_SafeNext == 1
    _P(snric)(&_BLL_this->SafeNext); /* TOOD but why we set it initially? */
  #elif BLL_set_SafeNext > 1
    _BLL_this->SafeNextCount = 0;
  #endif
}
_BLL_fdec(void, Close
){
  _BLL_fcall(_DestructAllNodes);
  #if BLL_set_StoreFormat == 0
    #if BLL_set_CPP_CopyAtPointerChange
      BLL_set_alloc_close(NodeList.ptr);
    #endif
    _P(_NodeList_Close)(&_BLL_this->NodeList);
  #elif BLL_set_StoreFormat == 1
    _BLL_fcall(_StoreFormat1_CloseAllocatedBlocks);
    _P(_BlockList_Close)(&_BLL_this->BlockList);
  #endif
}
#if BLL_set_CPP_ConstructDestruct
  public:
#endif

/* TOOD those 2 numbers in this function needs to be flexible */
_BLL_fdec(void, Clear
){
  _BLL_fcall(_DestructAllNodes);
  #if BLL_set_ResizeListAfterClear == 0
    #if BLL_set_StoreFormat == 0
      _BLL_this->NodeList.Current = 0;
    #elif BLL_set_StoreFormat == 1
      _BLL_fcall(_StoreFormat1_CloseAllocatedBlocks);
      _BLL_this->BlockList.Current = 0;
      _BLL_this->NodeCurrent = 0;
    #endif
  #else
    #if BLL_set_StoreFormat == 0
      _BLL_this->NodeList.Current = 0;
      #if BLL_set_CPP_CopyAtPointerChange
        /* TODO */
        __abort();
      #else
        _P(_NodeList_Reserve)(&_BLL_this->NodeList, 2);
      #endif
    #elif BLL_set_StoreFormat == 1
      _BLL_fcall(_StoreFormat1_CloseAllocatedBlocks);
      _P(_BlockList_ClearWithBuffer)(&_BLL_this->BlockList);
      _BLL_this->NodeCurrent = 0;
    #endif
  #endif

  _BLL_fcall(_AfterInitNodes);
}

/* TODO make implement of this with !BLL_set_LinkSentinel */
#if BLL_set_LinkSentinel
  _BLL_fdec(bool, IsNodeReferenceFronter,
    _P(NodeReference_t) srcnr,
    _P(NodeReference_t) dstnr
  ){
    do{
      _P(Node_t) *srcNode = _BLL_fcall(gln, srcnr);
      srcnr = srcNode->NextNodeReference;
      if(_P(inre)(srcnr, dstnr)){
        return 0;
      }
    }while(!_P(inre)(srcnr, _BLL_this->dst));
    return 1;
  }
#endif

#if BLL_set_SafeNext != 0
  _BLL_fdec(void, StartSafeNext,
    _P(NodeReference_t) nr
  ){
    #if BLL_set_SafeNext == 1
      _BLL_this->SafeNext = nr;
    #else
      _BLL_this->SafeNext[_BLL_this->SafeNextCount++] = nr;
    #endif
  }
  _BLL_fdec(_P(NodeReference_t), EndSafeNext
  ){
    _P(NodeReference_t) nr;
    #if BLL_set_SafeNext == 1
      nr = _BLL_this->SafeNext;
      _P(snric)(&_BLL_this->SafeNext); /* TODO but why we set? */
    #else
      nr = _BLL_this->SafeNext[--_BLL_this->SafeNextCount];
    #endif
    _P(Node_t) *Node = _BLL_fcall(GetNodeByReference, nr);
    return Node->NextNodeReference;
  }
  _BLL_fdec(_P(NodeReference_t), CheckSafeNext,
    uint8_t Depth
  ){
    ++Depth;
    #if BLL_set_SafeNext == 1
      return _BLL_this->SafeNext;
    #else
      return _BLL_this->SafeNext[_BLL_this->SafeNextCount - Depth];
    #endif
  }
#endif

#if BLL_set_Language == 1
  #if defined(BLL_set_MultipleType_Sizes)
    /* what syntax you would like */
  #elif defined(_BLL_HaveConstantNodeData)
    _P(NodeData_t) &operator[](_P(NodeReference_t) NR){
      return *GetNodeReferenceData(NR);
    }
  #else
    _P(NodeData_t) *operator[](_P(NodeReference_t) NR){
      return GetNodeReferenceData(NR);
    }
  #endif

  #if !BLL_set_Recycle && BLL_set_IntegerNR && !BLL_set_LinkSentinel
    _P(NodeData_t) *begin() { return &operator[](0); }
    _P(NodeData_t) *end() { return &operator[](Usage()); }
  #endif

  #if BLL_set_CPP_ConstructDestruct
    _P(t)(
      #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
        BLL_set_NodeSizeType NodeDataSize
      #endif
    ){
      Open(
        #if !defined(_BLL_HaveConstantNodeData) && !defined(BLL_set_MultipleType_Sizes)
          NodeDataSize
        #endif
      );
    }
    ~_P(t)(
    ){
      Close();
    }
  #endif

  #ifdef BLL_set_Overload_Declare
    BLL_set_Overload_Declare
  #endif

  BLL_StructEnd(_P(t))

  #if BLL_set_Link == 1
    static _P(NodeReference_t) _P(_NodeReference_Next)(_P(NodeReference_t) *nr, _P(t) *list){
      return list->gln(*nr)->NextNodeReference;
    }
    static _P(NodeReference_t) _P(_NodeReference_Prev)(_P(NodeReference_t) *nr, _P(t) *list){
      return list->gln(*nr)->PrevNodeReference;
    }
  #endif
#endif
