static
void _FED_RemoveLine(FED_t *wed, FED_LineReference_t LineReference){
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	_FED_CharacterList_close(&Line->CharacterList);
	_FED_LineList_unlink(&wed->LineList, LineReference);
}

static
void _FED_MoveCharacterToBeginOfLine(
	FED_t *wed,
	_FED_Line_t *srcLine, FED_LineReference_t srcLineReference,
	FED_CharacterReference_t srcCharacterReference, _FED_Character_t *srcCharacter,
	_FED_Line_t *dstLine, FED_LineReference_t dstLineReference
){
	FED_CharacterReference_t dstCharacterReference = _FED_CharacterList_NewNode(&dstLine->CharacterList);
	_FED_CharacterList_linkNext(&dstLine->CharacterList, dstLine->CharacterList.src, dstCharacterReference);
	_FED_CharacterList_Node_t *dstCharacterNode = _FED_CharacterList_GetNodeByReference(
		&dstLine->CharacterList,
		dstCharacterReference
	);
	_FED_Character_t *dstCharacter = &dstCharacterNode->data.data;
	if(srcCharacter->CursorReference != -1){
		_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(
			&wed->CursorList,
			srcCharacter->CursorReference
		);
		FED_Cursor_t *Cursor = &CursorNode->data.data;
		Cursor->FreeStyle.CharacterReference = dstCharacterReference;
		Cursor->FreeStyle.LineReference = dstLineReference;
	}
	srcLine->TotalWidth -= srcCharacter->width;
	dstLine->TotalWidth += srcCharacter->width;
	dstCharacter->CursorReference = srcCharacter->CursorReference;
	dstCharacter->width = srcCharacter->width;
	dstCharacter->data = srcCharacter->data;
	_FED_CharacterList_unlink(&srcLine->CharacterList, srcCharacterReference);
}
static
void _FED_MoveCharacterToEndOfLine(
	FED_t *wed,
	_FED_Line_t *srcLine, FED_LineReference_t srcLineReference,
	FED_CharacterReference_t srcCharacterReference, _FED_Character_t *srcCharacter,
	_FED_Line_t *dstLine, FED_LineReference_t dstLineReference
){
	FED_CharacterReference_t dstCharacterReference = _FED_CharacterList_NewNode(&dstLine->CharacterList);
	_FED_CharacterList_linkPrev(&dstLine->CharacterList, dstLine->CharacterList.dst, dstCharacterReference);
	_FED_CharacterList_Node_t *dstCharacterNode = _FED_CharacterList_GetNodeByReference(
		&dstLine->CharacterList,
		dstCharacterReference
	);
	_FED_Character_t *dstCharacter = &dstCharacterNode->data.data;
	if(srcCharacter->CursorReference != -1){
		_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(
			&wed->CursorList,
			srcCharacter->CursorReference
		);
		FED_Cursor_t *Cursor = &CursorNode->data.data;
		Cursor->FreeStyle.CharacterReference = dstCharacterReference;
		Cursor->FreeStyle.LineReference = dstLineReference;
	}
	srcLine->TotalWidth -= srcCharacter->width;
	dstLine->TotalWidth += srcCharacter->width;
	dstCharacter->CursorReference = srcCharacter->CursorReference;
	dstCharacter->width = srcCharacter->width;
	dstCharacter->data = srcCharacter->data;
	_FED_CharacterList_unlink(&srcLine->CharacterList, srcCharacterReference);
}

static
void _FED_MoveCursorFreeStyle(
	FED_t *wed,
	FED_CursorReference_t CursorReference,
	FED_Cursor_t *Cursor,
	FED_LineReference_t dstLineReference,
	FED_CharacterReference_t dstCharacterReference,
	_FED_Character_t *dstCharacter
){
	_FED_LineList_Node_t *srcLineNode = _FED_LineList_GetNodeByReference(
		&wed->LineList,
		Cursor->FreeStyle.LineReference
	);
	_FED_Line_t *srcLine = &srcLineNode->data.data;
	_FED_CharacterList_Node_t *srcCharacterNode = _FED_CharacterList_GetNodeByReference(
		&srcLine->CharacterList,
		Cursor->FreeStyle.CharacterReference
	);
	_FED_Character_t *srcCharacter = &srcCharacterNode->data.data;
	srcCharacter->CursorReference = -1;
	if(dstCharacter->CursorReference != -1){
		/* there is already cursor what should we do? */
		assert(0);
	}
	dstCharacter->CursorReference = CursorReference;
	Cursor->FreeStyle.LineReference = dstLineReference;
	Cursor->FreeStyle.CharacterReference = dstCharacterReference;
}
static
void _FED_MoveCursor_NoCleaning(
	FED_CursorReference_t CursorReference,
	FED_LineReference_t *srcLineReference, /* will be changed */
	FED_CharacterReference_t *srcCharacterReference, /* will be changed */
	FED_LineReference_t dstLineReference,
	FED_CharacterReference_t dstCharacterReference,
	_FED_Character_t *dstCharacter
){
	if(dstCharacter->CursorReference != -1){
		/* there is already cursor what should we do? */
		assert(0);
	}
	dstCharacter->CursorReference = CursorReference;
	*srcLineReference = dstLineReference;
	*srcCharacterReference = dstCharacterReference;
}
static
void _FED_MoveCursorSelection(
	FED_CursorReference_t CursorReference,
	FED_LineReference_t *srcLineReference, /* will be changed */
	FED_CharacterReference_t *srcCharacterReference, /* will be changed */
	_FED_Character_t *srcCharacter,
	FED_LineReference_t dstLineReference,
	FED_CharacterReference_t dstCharacterReference,
	_FED_Character_t *dstCharacter
){
	srcCharacter->CursorReference = -1;
	_FED_MoveCursor_NoCleaning(
		CursorReference,
		srcLineReference,
		srcCharacterReference,
		dstLineReference,
		dstCharacterReference,
		dstCharacter
	);
}

/* future implement that moves all cursors source to destination */
/* safe to call when source doesnt have cursor too */
static
void _FED_MoveAllCursors(
	FED_t *wed,
	FED_LineReference_t srcLineReference,
	FED_CharacterReference_t srcCharacterReference,
	_FED_Character_t *srcCharacter,
	FED_LineReference_t dstLineReference,
	FED_CharacterReference_t dstCharacterReference,
	_FED_Character_t *dstCharacter
){
	FED_CursorReference_t CursorReference = srcCharacter->CursorReference;
	if(CursorReference == -1){
		/* source doesnt have any cursor */
		return;
	}
	srcCharacter->CursorReference = -1;
	if(dstCharacter->CursorReference != -1){
		/* there is already cursor what should we do? */
		assert(0);
	}
	dstCharacter->CursorReference = CursorReference;
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			Cursor->FreeStyle.LineReference = dstLineReference;
			Cursor->FreeStyle.CharacterReference = dstCharacterReference;
			break;
		}
		case FED_CursorType_Selection_e:{
			if(
				Cursor->Selection.LineReference[0] == srcLineReference &&
				Cursor->Selection.CharacterReference[0] == srcCharacterReference
			){
				Cursor->Selection.LineReference[0] = dstLineReference;
				Cursor->Selection.CharacterReference[0] = dstCharacterReference;
			}
			else{
				Cursor->Selection.LineReference[1] = dstLineReference;
				Cursor->Selection.CharacterReference[1] = dstCharacterReference;
			}
			break;
		}
	}
}

static
void _FED_RemoveCharacter_Safe(
	FED_t *wed,
	FED_LineReference_t LineReference,
	_FED_Line_t *Line,
	FED_CharacterReference_t CharacterReference,
	_FED_Character_t *Character
){
	FED_CharacterReference_t dstCharacterReference = _FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		CharacterReference
	)->PrevNodeReference;
	_FED_CharacterList_Node_t *dstCharacterNode = _FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		dstCharacterReference
	);
	_FED_Character_t *dstCharacter = &dstCharacterNode->data.data;
	_FED_MoveAllCursors(wed, LineReference, CharacterReference, Character, LineReference, dstCharacterReference, dstCharacter);

	Line->TotalWidth -= Character->width;
	_FED_CharacterList_unlink(&Line->CharacterList, CharacterReference);
}
static
void _FED_RemoveCharacter_Unsafe(
	_FED_Line_t *Line,
	FED_CharacterReference_t CharacterReference,
	_FED_Character_t *Character
){
	Line->TotalWidth -= Character->width;
	_FED_CharacterList_unlink(&Line->CharacterList, CharacterReference);
}

/* returns 0 if left is possible */
static
bool _FED_GetLineAndCharacterOfLeft(
	FED_t *wed,
	FED_LineReference_t srcLineReference,
	_FED_LineList_Node_t *srcLineNode,
	FED_CharacterReference_t srcCharacterReference,
	FED_LineReference_t *dstLineReference,
	_FED_Line_t **dstLine,
	FED_CharacterReference_t *dstCharacterReference
){
	_FED_Line_t *srcLine = &srcLineNode->data.data;
	if(srcCharacterReference == srcLine->CharacterList.src){
		/* its begin of line. can we go up? */
		FED_LineReference_t PrevLineReference = srcLineNode->PrevNodeReference;
		if(PrevLineReference != wed->LineList.src){
			_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_FED_Line_t *PrevLine = &PrevLineNode->data.data;
			*dstLineReference = PrevLineReference;
			*dstLine = PrevLine;
			*dstCharacterReference = _FED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
			return 0;
		}
		else{
			/* we are already in top */
			return 1;
		}
	}
	else{
		*dstLineReference = srcLineReference;
		*dstLine = srcLine;
		*dstCharacterReference = _FED_CharacterList_GetNodeByReference(
			&srcLine->CharacterList,
			srcCharacterReference
		)->PrevNodeReference;
		return 0;
	}
}
static
void _FED_GetLineAndCharacterOfLeft_Unsafe(
	FED_t *wed,
	FED_LineReference_t srcLineReference,
	_FED_LineList_Node_t *srcLineNode,
	FED_CharacterReference_t srcCharacterReference,
	FED_LineReference_t *dstLineReference,
	_FED_Line_t **dstLine,
	FED_CharacterReference_t *dstCharacterReference
){
	_FED_Line_t *srcLine = &srcLineNode->data.data;
	if(srcCharacterReference == srcLine->CharacterList.src){
		/* its begin of line. can we go up? */
		FED_LineReference_t PrevLineReference = srcLineNode->PrevNodeReference;
		_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
		_FED_Line_t *PrevLine = &PrevLineNode->data.data;
		*dstLineReference = PrevLineReference;
		*dstLine = PrevLine;
		*dstCharacterReference = _FED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
	}
	else{
		*dstLineReference = srcLineReference;
		*dstLine = srcLine;
		*dstCharacterReference = _FED_CharacterList_GetNodeByReference(
			&srcLine->CharacterList,
			srcCharacterReference
		)->PrevNodeReference;
	}
}
/* returns 0 if right is possible */
static
bool _FED_GetLineAndCharacterOfRight(
	FED_t *wed,
	FED_LineReference_t srcLineReference,
	_FED_LineList_Node_t *srcLineNode,
	FED_CharacterReference_t srcCharacterReference,
	FED_LineReference_t *dstLineReference,
	_FED_Line_t **dstLine,
	FED_CharacterReference_t *dstCharacterReference
){
	_FED_Line_t *srcLine = &srcLineNode->data.data;
	FED_CharacterReference_t NextCharacterReference = _FED_CharacterList_GetNodeByReference(
		&srcLine->CharacterList,
		srcCharacterReference
	)->NextNodeReference;
	if(NextCharacterReference == srcLine->CharacterList.dst){
		/* its end of line. can we go up? */
		FED_LineReference_t NextLineReference = srcLineNode->NextNodeReference;
		if(NextLineReference != wed->LineList.dst){
			_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_FED_Line_t *NextLine = &NextLineNode->data.data;
			*dstLineReference = NextLineReference;
			*dstLine = NextLine;
			*dstCharacterReference = NextLine->CharacterList.src;
			return 0;
		}
		else{
			/* we are already in bottom */
			return 1;
		}
	}
	else{
		*dstLineReference = srcLineReference;
		*dstLine = srcLine;
		*dstCharacterReference = NextCharacterReference;
		return 0;
	}
}
static
void _FED_GetLineAndCharacterOfRight_Unsafe(
	FED_t *wed,
	FED_LineReference_t srcLineReference,
	_FED_LineList_Node_t *srcLineNode,
	FED_CharacterReference_t srcCharacterReference,
	FED_LineReference_t *dstLineReference,
	_FED_Line_t **dstLine,
	FED_CharacterReference_t *dstCharacterReference
){
	_FED_Line_t *srcLine = &srcLineNode->data.data;
	FED_CharacterReference_t NextCharacterReference = _FED_CharacterList_GetNodeByReference(
		&srcLine->CharacterList,
		srcCharacterReference
	)->NextNodeReference;
	if(NextCharacterReference == srcLine->CharacterList.dst){
		/* its end of line */
		FED_LineReference_t NextLineReference = srcLineNode->NextNodeReference;
		_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_FED_Line_t *NextLine = &NextLineNode->data.data;
		*dstLineReference = NextLineReference;
		*dstLine = NextLine;
		*dstCharacterReference = NextLine->CharacterList.src;
	}
	else{
		*dstLineReference = srcLineReference;
		*dstLine = srcLine;
		*dstCharacterReference = NextCharacterReference;
	}
}

/* this function able to change wed->LineList pointers */
/* returns 0 if success */
static
bool _FED_OpenExtraLine(
	FED_t *wed,
	FED_LineReference_t LineReference,
	FED_LineReference_t *NextLineReference,
	_FED_LineList_Node_t **NextLineNode
){
	if(_FED_LineList_usage(&wed->LineList) == wed->LineLimit){
		if(_FED_LineList_GetNodeLast(&wed->LineList) == LineReference){
			return 1;
		}
	}
	*NextLineReference = _FED_LineList_NewNode(&wed->LineList);
	_FED_LineList_linkNext(&wed->LineList, LineReference, *NextLineReference);
	*NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *NextLineReference);
	_FED_Line_t *NextLine = &(*NextLineNode)->data.data;
	NextLine->TotalWidth = 0;
	_FED_Line_t *Line = &_FED_LineList_GetNodeByReference(&wed->LineList, LineReference)->data.data;
	if(Line->IsEndLine){
		Line->IsEndLine = 0;
		NextLine->IsEndLine = 1;
	}
	else{
		NextLine->IsEndLine = 0;
	}
	_FED_CharacterList_open(&NextLine->CharacterList);
	_FED_CharacterList_Node_t *GodCharacterNode = _FED_CharacterList_GetNodeByReference(
		&NextLine->CharacterList,
		NextLine->CharacterList.src
	);
	_FED_Character_t *GodCharacter = &GodCharacterNode->data.data;
	GodCharacter->CursorReference = -1;
	if(_FED_LineList_usage(&wed->LineList) > wed->LineLimit){
		FED_LineReference_t LastLineReference = _FED_LineList_GetNodeLast(&wed->LineList);
		_FED_LineList_Node_t *LastLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LastLineReference);
		_FED_Line_t *LastLine = &LastLineNode->data.data;
		FED_LineReference_t LastPrevLineReference = LastLineNode->PrevNodeReference;
		_FED_LineList_Node_t *LastPrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LastPrevLineReference);
		_FED_Line_t *LastPrevLine = &LastPrevLineNode->data.data;

		/* no need to check is it had earlier or not */
		LastPrevLine->IsEndLine = 1;

		FED_CharacterReference_t LastPrevLineLastCharacterReference = _FED_CharacterList_GetNodeLast(&LastPrevLine->CharacterList);
		_FED_CharacterList_Node_t *LastPrevLineLastCharacterNode = _FED_CharacterList_GetNodeByReference(&LastPrevLine->CharacterList, LastPrevLineLastCharacterReference);
		_FED_Character_t *LastPrevLineLastCharacter = &LastPrevLineLastCharacterNode->data.data;
		FED_CharacterReference_t LastLineCharacterReference = LastLine->CharacterList.src;
		while(LastLineCharacterReference != LastLine->CharacterList.dst){
			_FED_CharacterList_Node_t *LastLineCharacterNode = _FED_CharacterList_GetNodeByReference(&LastLine->CharacterList, LastLineCharacterReference);
			_FED_Character_t *LastLineCharacter = &LastLineCharacterNode->data.data;
			_FED_MoveAllCursors(wed, LastLineReference, LastLineCharacterReference, LastLineCharacter, LastPrevLineReference, LastPrevLineLastCharacterReference, LastPrevLineLastCharacter);
			LastLineCharacterReference = LastLineCharacterNode->NextNodeReference;
		}
		_FED_RemoveLine(wed, LastLineReference);
	}
	return 0;
}

static
bool _FED_IsLineMembersFit(FED_t *wed, uint32_t CharacterAmount, uint32_t WidthAmount){
	if(CharacterAmount > wed->LineCharacterLimit){
		return 0;
	}
	if(WidthAmount > wed->LineWidth){
		return 0;
	}
	return 1;
}
static
bool _FED_IsLineFit(
	FED_t *wed,
	_FED_Line_t *Line
){
	return _FED_IsLineMembersFit(wed, _FED_CharacterList_usage(&Line->CharacterList), Line->TotalWidth);
}
static
bool _FED_CanCharacterFitLine(
	FED_t *wed,
	_FED_Line_t *Line,
	_FED_Character_t *Character
){
	return _FED_IsLineMembersFit(wed, _FED_CharacterList_usage(&Line->CharacterList) + 1, Line->TotalWidth + Character->width);
}
static
FED_CharacterReference_t _FED_LastCharacterReferenceThatFitsToLine(FED_t *wed, FED_LineReference_t LineReference){
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	uint32_t CharacterAmount = _FED_CharacterList_usage(&Line->CharacterList);
	uint32_t WidthAmount = Line->TotalWidth;
	FED_CharacterReference_t CharacterReference = _FED_CharacterList_GetNodeLast(&Line->CharacterList);
	while(1){
		if(CharacterReference == Line->CharacterList.src){
			return CharacterReference;
		}
		_FED_CharacterList_Node_t *CharacterNode = _FED_CharacterList_GetNodeByReference(&Line->CharacterList, CharacterReference);
		_FED_Character_t *Character = &CharacterNode->data.data;
		if(_FED_IsLineMembersFit(wed, CharacterAmount, WidthAmount)){
			return CharacterReference;
		}
		CharacterAmount -= 1;
		WidthAmount -= Character->width;
		CharacterReference = CharacterNode->PrevNodeReference;
	}
}

static
void _FED_SlideToNext(FED_t *wed, FED_LineReference_t LineReference, _FED_LineList_Node_t *LineNode){
	begin:
	FED_LineReference_t NextLineReference;
	_FED_LineList_Node_t *NextLineNode;
	bool IsLoopEntered = !_FED_IsLineFit(wed, &LineNode->data.data);
	while(!_FED_IsLineFit(wed, &LineNode->data.data)){
		if(LineNode->data.data.IsEndLine){
			/* if line has endline we need to create new line to slide */
			if(_FED_OpenExtraLine(wed, LineReference, &NextLineReference, &NextLineNode)){
				FED_CharacterReference_t dstCharacterReference = _FED_LastCharacterReferenceThatFitsToLine(wed, LineReference);
				_FED_Character_t *dstCharacter = &_FED_CharacterList_GetNodeByReference(&LineNode->data.data.CharacterList, dstCharacterReference)->data.data;
				FED_CharacterReference_t srcCharacterReference = _FED_CharacterList_GetNodeLast(&LineNode->data.data.CharacterList);
				do{
					_FED_CharacterList_Node_t *srcCharacterNode = _FED_CharacterList_GetNodeByReference(&LineNode->data.data.CharacterList, srcCharacterReference);
					_FED_MoveAllCursors(wed, LineReference, srcCharacterReference, &srcCharacterNode->data.data, LineReference, dstCharacterReference, dstCharacter);
					FED_CharacterReference_t srcCharacterReference_temp = srcCharacterNode->PrevNodeReference;
					_FED_RemoveCharacter_Unsafe(&LineNode->data.data, srcCharacterReference, &srcCharacterNode->data.data);
					srcCharacterReference = srcCharacterReference_temp;
				}while(srcCharacterReference != dstCharacterReference);
				return;
			}
			/* that function maybe changed line address so lets get it again */
			LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		}
		else{
			NextLineReference = LineNode->NextNodeReference;
			NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		}
		FED_CharacterReference_t CharacterReference = _FED_CharacterList_GetNodeLast(&LineNode->data.data.CharacterList);
		_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
			&LineNode->data.data.CharacterList,
			CharacterReference
		)->data.data;
		_FED_MoveCharacterToBeginOfLine(
			wed,
			&LineNode->data.data,
			LineReference,
			CharacterReference,
			Character,
			&NextLineNode->data.data,
			NextLineReference
		);
	}
	if(IsLoopEntered){
		LineReference = NextLineReference;
		LineNode = NextLineNode;
		goto begin;
	}
}

static
void _FED_LineSlideBackFromNext(
	FED_t *wed,
	FED_LineReference_t LineReference,
	_FED_LineList_Node_t *LineNode,
	FED_LineReference_t NextLineReference,
	_FED_LineList_Node_t *NextLineNode
){
	Begin:
	FED_CharacterReference_t NextCharacterReference = _FED_CharacterList_GetNodeFirst(&NextLineNode->data.data.CharacterList);
	if(NextCharacterReference == NextLineNode->data.data.CharacterList.dst){
		FED_CharacterReference_t LastCharacterReference = _FED_CharacterList_GetNodeLast(&LineNode->data.data.CharacterList);
		_FED_CharacterList_Node_t *LastCharacterNode = _FED_CharacterList_GetNodeByReference(&LineNode->data.data.CharacterList, LastCharacterReference);
		_FED_Character_t *LastCharacter = &LastCharacterNode->data.data;
		NextCharacterReference = NextLineNode->data.data.CharacterList.src;
		_FED_CharacterList_Node_t *NextCharacterNode = _FED_CharacterList_GetNodeByReference(&NextLineNode->data.data.CharacterList, NextCharacterReference);
		_FED_Character_t *NextCharacter = &NextCharacterNode->data.data;
		_FED_MoveAllCursors(wed, NextLineReference, NextCharacterReference, NextCharacter, LineReference, LastCharacterReference, LastCharacter);
		bool IsEndLine = NextLineNode->data.data.IsEndLine;
		_FED_RemoveLine(wed, NextLineReference);
		if(NextLineNode->data.data.IsEndLine){
			LineNode->data.data.IsEndLine = 1;
			return;
		}
		else{
			NextLineReference = LineNode->NextNodeReference;
			NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			goto Begin;
		}
	}
	_FED_CharacterList_Node_t *NextCharacterNode = _FED_CharacterList_GetNodeByReference(&NextLineNode->data.data.CharacterList, NextCharacterReference);
	_FED_Character_t *NextCharacter = &NextCharacterNode->data.data;
	if(_FED_CanCharacterFitLine(wed, &LineNode->data.data, NextCharacter)){
		_FED_MoveCharacterToEndOfLine(wed, &NextLineNode->data.data, NextLineReference, NextCharacterReference, NextCharacter, &LineNode->data.data, LineReference);

		/* TODO this only needed to be processed per line */
		FED_CharacterReference_t LastCharacterReference = _FED_CharacterList_GetNodeLast(&LineNode->data.data.CharacterList);
		_FED_CharacterList_Node_t *LastCharacterNode = _FED_CharacterList_GetNodeByReference(&LineNode->data.data.CharacterList, LastCharacterReference);
		_FED_Character_t *LastCharacter = &LastCharacterNode->data.data;
		NextCharacterReference = NextLineNode->data.data.CharacterList.src;
		NextCharacterNode = _FED_CharacterList_GetNodeByReference(&NextLineNode->data.data.CharacterList, NextCharacterReference);
		NextCharacter = &NextCharacterNode->data.data;
		_FED_MoveAllCursors(wed, NextLineReference, NextCharacterReference, NextCharacter, LineReference, LastCharacterReference, LastCharacter);

		goto Begin;
	}
	if(NextLineNode->data.data.IsEndLine){
		return;
	}
	LineReference = NextLineReference;
	LineNode = NextLineNode;
	NextLineReference = LineNode->NextNodeReference;
	NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
	goto Begin;
}

static
void _FED_LineIsDecreased(FED_t *wed, FED_LineReference_t LineReference, _FED_LineList_Node_t *LineNode){
	FED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
	if(PrevLineReference != wed->LineList.src){
		_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
		if(PrevLineNode->data.data.IsEndLine){
			goto ToNext;
		}
		_FED_LineSlideBackFromNext(wed, PrevLineReference, PrevLineNode, LineReference, LineNode);
	}
	else{
		ToNext:
		if(LineNode->data.data.IsEndLine){
			return;
		}
		FED_LineReference_t NextLineReference = LineNode->NextNodeReference;
		_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_FED_LineSlideBackFromNext(wed, LineReference, LineNode, NextLineReference, NextLineNode);
	}
}

static
void _FED_LineIsIncreased(FED_t *wed, FED_LineReference_t LineReference, _FED_LineList_Node_t *LineNode){
	_FED_LineIsDecreased(wed, LineReference, LineNode);
	_FED_SlideToNext(wed, LineReference, LineNode);
}

static
void _FED_CursorIsTriggered(FED_Cursor_t *Cursor){
	/* this function must be called when something is changed or could change */
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			Cursor->FreeStyle.PreferredWidth = -1;
			break;
		}
		case FED_CursorType_Selection_e:{
			Cursor->Selection.PreferredWidth = -1;
			break;
		}
	}
}

static
void _FED_GetCharacterFromLineByWidth(
	FED_t *wed,
	_FED_Line_t *Line,
	uint32_t Width,
	FED_CharacterReference_t *pCharacterReference,
	_FED_Character_t **pCharacter
){
	uint32_t iWidth = 0;
	FED_CharacterReference_t CharacterReference = Line->CharacterList.src;
	while(1){
		FED_CharacterReference_t NextCharacterReference = _FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->NextNodeReference;
		if(NextCharacterReference == Line->CharacterList.dst){
			/* lets return Character */
			*pCharacter = &_FED_CharacterList_GetNodeByReference(&Line->CharacterList, CharacterReference)->data.data;
			*pCharacterReference = CharacterReference;
			return;
		}
		else{
			_FED_Character_t *NextCharacter = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				NextCharacterReference
			)->data.data;
			if((iWidth + NextCharacter->width) >= Width){
				/* we need to return Character or NextCharacter depends about how close to Width */
				uint32_t CurrentDiff = Width - iWidth;
				uint32_t NextDiff = iWidth + NextCharacter->width - Width;
				if(CurrentDiff <= NextDiff){
					/* lets return Character */
					*pCharacter = &_FED_CharacterList_GetNodeByReference(
						&Line->CharacterList,
						CharacterReference
					)->data.data;
					*pCharacterReference = CharacterReference;
					return;
				}
				else{
					/* lets return NextCharacter */
					*pCharacter = NextCharacter;
					*pCharacterReference = NextCharacterReference;
					return;
				}
			}
			else{
				/* lets loop more */
				iWidth += NextCharacter->width;
				CharacterReference = NextCharacterReference;
			}
		}
	}
}

static
uint32_t _FED_CalculatePositionOfCharacterInLine(FED_t *wed, _FED_Line_t *Line, FED_CharacterReference_t pCharacterReference){
	uint32_t Width = 0;
	FED_CharacterReference_t CharacterReference = _FED_CharacterList_GetNodeFirst(&Line->CharacterList);
	while(1){
		_FED_CharacterList_Node_t *CharacterNode = _FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		);
		_FED_Character_t *Character = &CharacterNode->data.data;
		Width += Character->width;
		if(CharacterReference == pCharacterReference){
			break;
		}
		CharacterReference = CharacterNode->NextNodeReference;
	}
	return Width;
}

static
void _FED_CalculatePreferredWidthIfNeeded(
	FED_t *wed,
	_FED_Line_t *Line,
	FED_CharacterReference_t CharacterReference,
	uint32_t *PreferredWidth
){
	if(*PreferredWidth != -1){
		/* no need */
		return;
	}
	if(CharacterReference == Line->CharacterList.src){
		/* cursor is in begin so PreferredWidth must be 0 */
		*PreferredWidth = 0;
	}
	else{
		*PreferredWidth = _FED_CalculatePositionOfCharacterInLine(
			wed,
			Line,
			CharacterReference
		);
	}
}

static
void _FED_CursorConvertFreeStyleToSelection(
	FED_CursorReference_t CursorReference,
	FED_Cursor_t *Cursor,
	FED_LineReference_t LineReference,
	_FED_Line_t *Line,
	FED_CharacterReference_t CharacterReference,
	_FED_Character_t *Character,
	uint32_t PreferredWidth
){
	Cursor->type = FED_CursorType_Selection_e;
	Cursor->Selection.PreferredWidth = PreferredWidth;
	Cursor->Selection.LineReference[0] = Cursor->FreeStyle.LineReference;
	Cursor->Selection.CharacterReference[0] = Cursor->FreeStyle.CharacterReference;
	Cursor->Selection.LineReference[1] = LineReference;
	Cursor->Selection.CharacterReference[1] = CharacterReference;
	if(Character->CursorReference != -1){
		/* what will happen to cursor? */
		assert(0);
	}
	Character->CursorReference = CursorReference;
}

static
void _FED_CursorConvertSelectionToFreeStyle(FED_t *wed, FED_Cursor_t *Cursor, bool Direction){
	Cursor->type = FED_CursorType_FreeStyle_e;
	{
		FED_LineReference_t LineReference = Cursor->Selection.LineReference[Direction ^ 1];
		_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_FED_Line_t *Line = &LineNode->data.data;
		FED_CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction ^ 1];
		_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->data.data;
		Character->CursorReference = -1;
	}
	FED_LineReference_t LineReference = Cursor->Selection.LineReference[Direction];
	FED_CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction];
	uint32_t PreferredWidth = Cursor->Selection.PreferredWidth;
	Cursor->FreeStyle.LineReference = LineReference;
	Cursor->FreeStyle.CharacterReference = CharacterReference;
	Cursor->FreeStyle.PreferredWidth = PreferredWidth;
}

static
void _FED_CursorDeleteSelectedAndMakeCursorFreeStyle(FED_t *wed, FED_Cursor_t *Cursor){
	bool Direction;
	FED_LineReference_t LineReference0;
	FED_LineReference_t LineReference;
	if(Cursor->Selection.LineReference[0] != Cursor->Selection.LineReference[1]){
		/* lets compare lines */
		if(_FED_LineList_IsNodeReferenceFronter(
			&wed->LineList,
			Cursor->Selection.LineReference[0],
			Cursor->Selection.LineReference[1])
		){
			LineReference0 = Cursor->Selection.LineReference[1];
			LineReference = Cursor->Selection.LineReference[0];
			Direction = 1;
		}
		else{
			LineReference0 = Cursor->Selection.LineReference[0];
			LineReference = Cursor->Selection.LineReference[1];
			Direction = 0;
		}
	}
	else{
		/* lets compare characters */
		LineReference0 = Cursor->Selection.LineReference[0];
		LineReference = Cursor->Selection.LineReference[0];
		_FED_Line_t *Line = &_FED_LineList_GetNodeByReference(
			&wed->LineList,
			Cursor->Selection.LineReference[0]
		)->data.data;
		if(_FED_CharacterList_IsNodeReferenceFronter(
			&Line->CharacterList,
			Cursor->Selection.CharacterReference[0],
			Cursor->Selection.CharacterReference[1])
		){
			Direction = 1;
		}
		else{
			Direction = 0;
		}
	}
	FED_CharacterReference_t CharacterReference0 = Cursor->Selection.CharacterReference[Direction];
	FED_CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction ^ 1];
	_FED_LineList_Node_t *LineNode0 = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference0);
	_FED_Line_t *Line0 = &LineNode0->data.data;
	_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, Direction);
	while(1){
		_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_FED_Line_t *Line = &LineNode->data.data;
		_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->data.data;
		_FED_Character_t *Character0 = &_FED_CharacterList_GetNodeByReference(
			&Line0->CharacterList,
			CharacterReference0
		)->data.data;
		_FED_MoveAllCursors(
			wed,
			LineReference,
			CharacterReference,
			Character,
			LineReference0,
			CharacterReference0,
			Character0
		);
		FED_LineReference_t dstLineReference;
		_FED_Line_t *dstLine;
		FED_CharacterReference_t dstCharacterReference;
		if(CharacterReference != Line->CharacterList.src){
			_FED_GetLineAndCharacterOfLeft_Unsafe(
				wed,
				LineReference,
				LineNode,
				CharacterReference,
				&dstLineReference,
				&dstLine,
				&dstCharacterReference
			);
			_FED_RemoveCharacter_Unsafe(Line, CharacterReference, Character);
			if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
				/* we did reach where we go */
				_FED_LineIsDecreased(wed, LineReference, LineNode);
				return;
			}
		}
		else{
			/* we need to delete something from previous line */
			BeginOfCharacterReferenceIsFirst:
			dstLineReference = LineNode->PrevNodeReference;
			_FED_LineList_Node_t *dstLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, dstLineReference);
			dstLine = &dstLineNode->data.data;
			dstCharacterReference = _FED_CharacterList_GetNodeLast(&dstLine->CharacterList);
			if(dstLine->IsEndLine == 1){
				if(_FED_CharacterList_usage(&dstLine->CharacterList) == 0){
					/* lets delete line */
					_FED_Character_t *srcGodCharacter = &_FED_CharacterList_GetNodeByReference(
						&Line->CharacterList,
						Line->CharacterList.src
					)->data.data;
					_FED_Character_t *dstGodCharacter = &_FED_CharacterList_GetNodeByReference(
						&dstLine->CharacterList,
						dstLine->CharacterList.src
					)->data.data;
					_FED_MoveAllCursors(
						wed,
						dstLineReference,
						dstCharacterReference,
						dstGodCharacter,
						LineReference,
						CharacterReference,
						srcGodCharacter
					);
					_FED_RemoveLine(wed, dstLineReference);
					if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
						/* we did reach where we go */
						_FED_LineIsDecreased(wed, LineReference, LineNode);
						return;
					}
					goto BeginOfCharacterReferenceIsFirst;
				}
				else{
					dstLine->IsEndLine = 0;
					if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
						/* we did reach where we go */
						_FED_LineIsDecreased(wed, LineReference, LineNode);
						return;
					}
				}
			}
			else{
				/* nothing to delete */
				if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
					/* we did reach where we go */
					_FED_LineIsDecreased(wed, LineReference, LineNode);
					return;
				}
			}
		}
		if(LineReference != dstLineReference){
			/* we got other line */
			LineReference = dstLineReference;
		}
		CharacterReference = dstCharacterReference;
	}
}
