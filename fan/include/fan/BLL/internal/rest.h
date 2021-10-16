#if BLL_set_PadNode == 0
	#pragma pack(push, 1)
#endif
typedef struct{
	#if BLL_set_PreferNextFirst == 1
		CONCAT2(BLL_set_prefix, _NodeReference_t) NextNodeReference;
		CONCAT2(BLL_set_prefix, _NodeReference_t) PrevNodeReference;
	#else
		CONCAT2(BLL_set_prefix, _NodeReference_t) PrevNodeReference;
		CONCAT2(BLL_set_prefix, _NodeReference_t) NextNodeReference;
	#endif
	struct{
		BLL_set_node_data
	}data;
}CONCAT2(BLL_set_prefix, _Node_t);
#if BLL_set_PadNode == 0
	#pragma pack(pop)
#endif

static
bool
CONCAT2(BLL_set_prefix, _IsNodeReferenceInvalid)
(CONCAT2(BLL_set_prefix, _t) *list, CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference){
	return NodeReference >= list->nodes.Current;
}

#if BLL_set_IsNodeUnlinked == 1
	static
	bool
	CONCAT2(BLL_set_prefix, _IsNodeUnlinked)
	(CONCAT2(BLL_set_prefix, _t) *list, CONCAT2(BLL_set_prefix, _Node_t) *Node){
		if(Node->PrevNodeReference == -1){
			return 1;
		}
		return 0;
	}
	static
	bool
	CONCAT2(BLL_set_prefix, _IsNodeReferenceUnlinked)
	(CONCAT2(BLL_set_prefix, _t) *list, CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference){
		CONCAT2(BLL_set_prefix, _Node_t) *Node = &((CONCAT2(BLL_set_prefix, _Node_t) *)&list->nodes.ptr[0])[NodeReference];
		return CONCAT2(BLL_set_prefix, _IsNodeUnlinked)(list, Node);
	}
#endif

static
CONCAT2(BLL_set_prefix, _Node_t) *
CONCAT3(_, BLL_set_prefix, _GetNodeByReference)
(CONCAT2(BLL_set_prefix, _t) *list, CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference){
	return &((CONCAT2(BLL_set_prefix, _Node_t) *)&list->nodes.ptr[0])[NodeReference];
}

static
CONCAT2(BLL_set_prefix, _Node_t) *
CONCAT2(BLL_set_prefix, _GetNodeByReference)
(CONCAT2(BLL_set_prefix, _t) *list, CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference){
	#if BLL_set_debug_InvalidAction == 1
		if(NodeReference >= list->nodes.Current){
			assert(0);
		}
	#endif
	CONCAT2(BLL_set_prefix, _Node_t) *Node = CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, NodeReference);
	#if BLL_set_debug_InvalidAction == 1
		do{
			#if BLL_set_debug_InvalidAction_srcAccess == 0
				if(NodeReference == list->src){
					break;
				}
			#endif
			#if BLL_set_debug_InvalidAction_dstAccess == 0
				if(NodeReference == list->dst){
					break;
				}
			#endif
			if(CONCAT2(BLL_set_prefix, _IsNodeUnlinked)(list, Node)){
				assert(0);
			}
		}while(0);
	#endif
	return Node;
}

static
uint_t
CONCAT2(BLL_set_prefix, _usage)
(CONCAT2(BLL_set_prefix, _t) *list){
	return list->nodes.Current - list->e.p - 2;
}

static
void
CONCAT2(BLL_set_prefix, _open)
(CONCAT2(BLL_set_prefix, _t) *list){
	fan::VEC_init(&list->nodes, sizeof(CONCAT2(BLL_set_prefix, _Node_t)));
	list->e.c = 0;
	list->e.p = 0;
	VEC_handle0(&list->nodes, 2);
	list->src = 0;
	list->dst = 1;

	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, list->src)->NextNodeReference = list->dst;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, list->dst)->PrevNodeReference = list->src;
}
static
void
CONCAT2(BLL_set_prefix, _close)
(CONCAT2(BLL_set_prefix, _t) *list){
	fan::VEC_free(&list->nodes);
}

static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNode_empty)
(CONCAT2(BLL_set_prefix, _t) *list){
	CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference = list->e.c;
	list->e.c = CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, NodeReference)->NextNodeReference;
	list->e.p--;
	return NodeReference;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNode_alloc)
(CONCAT2(BLL_set_prefix, _t) *list){
	VEC_handle(&list->nodes);
	return list->nodes.Current++;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNode)
(CONCAT2(BLL_set_prefix, _t) *list){
	if(list->e.p){
		return CONCAT2(BLL_set_prefix, _NewNode_empty)(list);
	}
	else{
		return CONCAT2(BLL_set_prefix, _NewNode_alloc)(list);
	}
}

static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNodeFirst_empty)
(CONCAT2(BLL_set_prefix, _t) *list){
	CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference = CONCAT2(BLL_set_prefix, _NewNode_empty)(list);
	CONCAT2(BLL_set_prefix, _NodeReference_t) srcNodeReference = list->src;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, NodeReference)->NextNodeReference = srcNodeReference;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, srcNodeReference)->PrevNodeReference = NodeReference;
	list->src = NodeReference;
	return srcNodeReference;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNodeFirst_alloc)
(CONCAT2(BLL_set_prefix, _t) *list){
	CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference = CONCAT2(BLL_set_prefix, _NewNode_alloc)(list);
	CONCAT2(BLL_set_prefix, _NodeReference_t) srcNodeReference = list->src;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, NodeReference)->NextNodeReference = srcNodeReference;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, srcNodeReference)->PrevNodeReference = NodeReference;
	list->src = NodeReference;
	return srcNodeReference;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNodeFirst)
(CONCAT2(BLL_set_prefix, _t) *list){
	if(list->e.p){
		return CONCAT2(BLL_set_prefix, _NewNodeFirst_empty)(list);
	}
	else{
		return CONCAT2(BLL_set_prefix, _NewNodeFirst_alloc)(list);
	}
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNodeLast_empty)
(CONCAT2(BLL_set_prefix, _t) *list){
	CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference = CONCAT2(BLL_set_prefix, _NewNode_empty)(list);
	CONCAT2(BLL_set_prefix, _NodeReference_t) dstNodeReference = list->dst;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, NodeReference)->PrevNodeReference = dstNodeReference;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, dstNodeReference)->NextNodeReference = NodeReference;
	#if BLL_set_debug_InvalidAction == 1
		CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, dstNodeReference)->PrevNodeReference = 0;
	#endif
	list->dst = NodeReference;
	return dstNodeReference;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNodeLast_alloc)
(CONCAT2(BLL_set_prefix, _t) *list){
	CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference = CONCAT2(BLL_set_prefix, _NewNode_alloc)(list);
	CONCAT2(BLL_set_prefix, _NodeReference_t) dstNodeReference = list->dst;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, NodeReference)->PrevNodeReference = dstNodeReference;
	CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, dstNodeReference)->NextNodeReference = NodeReference;
	#if BLL_set_debug_InvalidAction == 1
		CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, dstNodeReference)->PrevNodeReference = 0;
	#endif
	list->dst = NodeReference;
	return dstNodeReference;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _NewNodeLast)
(CONCAT2(BLL_set_prefix, _t) *list){
	if(list->e.p){
		return CONCAT2(BLL_set_prefix, _NewNodeLast_empty)(list);
	}
	else{
		return CONCAT2(BLL_set_prefix, _NewNodeLast_alloc)(list);
	}
}

static
void
CONCAT2(BLL_set_prefix, _linkNext)
(
	CONCAT2(BLL_set_prefix, _t) *list,
	CONCAT2(BLL_set_prefix, _NodeReference_t) srcNodeReference,
	CONCAT2(BLL_set_prefix, _NodeReference_t) dstNodeReference
){
	CONCAT2(BLL_set_prefix, _Node_t) *srcNode = CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, srcNodeReference);
	CONCAT2(BLL_set_prefix, _Node_t) *dstNode = CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, dstNodeReference);
	CONCAT2(BLL_set_prefix, _NodeReference_t) nextNodeReference = srcNode->NextNodeReference;
	CONCAT2(BLL_set_prefix, _Node_t) *nextNode = CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, nextNodeReference);
	srcNode->NextNodeReference = dstNodeReference;
	dstNode->PrevNodeReference = srcNodeReference;
	dstNode->NextNodeReference = nextNodeReference;
	nextNode->PrevNodeReference = dstNodeReference;
}
static
void
CONCAT2(BLL_set_prefix, _linkPrev)
(
	CONCAT2(BLL_set_prefix, _t) *list,
	CONCAT2(BLL_set_prefix, _NodeReference_t) srcNodeReference,
	CONCAT2(BLL_set_prefix, _NodeReference_t) dstNodeReference
){
	CONCAT2(BLL_set_prefix, _Node_t) *srcNode = CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, srcNodeReference);
	CONCAT2(BLL_set_prefix, _NodeReference_t) prevNodeReference = srcNode->PrevNodeReference;
	CONCAT2(BLL_set_prefix, _Node_t) *prevNode = CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, prevNodeReference);
	prevNode->NextNodeReference = dstNodeReference;
	CONCAT2(BLL_set_prefix, _Node_t) *dstNode = CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, dstNodeReference);
	dstNode->PrevNodeReference = prevNodeReference;
	dstNode->NextNodeReference = srcNodeReference;
	srcNode->PrevNodeReference = dstNodeReference;
}

static
void
CONCAT2(BLL_set_prefix, _unlink)
(CONCAT2(BLL_set_prefix, _t) *list, CONCAT2(BLL_set_prefix, _NodeReference_t) NodeReference){
	#if BLL_set_debug_InvalidAction == 1
		assert(NodeReference != list->src);
		assert(NodeReference != list->dst);
	#endif
	CONCAT2(BLL_set_prefix, _Node_t) *Node = CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, NodeReference);
	CONCAT2(BLL_set_prefix, _NodeReference_t) nextNodeReference = Node->NextNodeReference;
	CONCAT2(BLL_set_prefix, _NodeReference_t) prevNodeReference = Node->PrevNodeReference;
	CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, prevNodeReference)->NextNodeReference = nextNodeReference;
	CONCAT2(BLL_set_prefix, _GetNodeByReference)(list, nextNodeReference)->PrevNodeReference = prevNodeReference;

	Node->NextNodeReference = list->e.c;
	Node->PrevNodeReference = -1;
	list->e.c = NodeReference;
	list->e.p++;
}

static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _GetNodeFirst)
(CONCAT2(BLL_set_prefix, _t) *list){
	return CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, list->src)->NextNodeReference;
}
static
CONCAT2(BLL_set_prefix, _NodeReference_t)
CONCAT2(BLL_set_prefix, _GetNodeLast)
(CONCAT2(BLL_set_prefix, _t) *list){
	return CONCAT3(_, BLL_set_prefix, _GetNodeByReference)(list, list->dst)->PrevNodeReference;
}

static
bool
CONCAT2(BLL_set_prefix, _IsNodeReferenceFronter)
(
	CONCAT2(BLL_set_prefix, _t) *list,
	CONCAT2(BLL_set_prefix, _NodeReference_t) srcNodeReference,
	CONCAT2(BLL_set_prefix, _NodeReference_t) dstNodeReference
){
	do{
		CONCAT2(BLL_set_prefix, _Node_t) *srcNode = CONCAT2(BLL_set_prefix, _GetNodeByReference)(
			list,
			srcNodeReference
		);
		srcNodeReference = srcNode->NextNodeReference;
		if(srcNodeReference == dstNodeReference){
			return 0;
		}
	}while(srcNodeReference != list->dst);
	return 1;
}
