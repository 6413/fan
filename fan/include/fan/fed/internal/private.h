void _WED_MoveCharacterToBeginOfLine(
	WED_t *wed,
	_WED_Line_t *srcLine, WED_LineReference_t srcLineReference,
	WED_CharacterReference_t srcCharacterReference, _WED_Character_t *srcCharacter,
	_WED_Line_t *dstLine, WED_LineReference_t dstLineReference
){
	WED_CharacterReference_t dstCharacterReference = _WED_CharacterList_NewNode(&dstLine->CharacterList);
	_WED_CharacterList_linkNext(&dstLine->CharacterList, dstLine->CharacterList.src, dstCharacterReference);
	_WED_CharacterList_Node_t *dstCharacterNode = _WED_CharacterList_GetNodeByReference(
		&dstLine->CharacterList,
		dstCharacterReference
	);
	_WED_Character_t *dstCharacter = &dstCharacterNode->data.data;
	if(srcCharacter->CursorReference != -1){
		_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(
			&wed->CursorList,
			srcCharacter->CursorReference
		);
		WED_Cursor_t *Cursor = &CursorNode->data.data;
		Cursor->FreeStyle.CharacterReference = dstCharacterReference;
		Cursor->FreeStyle.LineReference = dstLineReference;
	}
	srcLine->TotalWidth -= srcCharacter->width;
	dstLine->TotalWidth += srcCharacter->width;
	dstCharacter->CursorReference = srcCharacter->CursorReference;
	dstCharacter->width = srcCharacter->width;
	dstCharacter->data = srcCharacter->data;
	_WED_CharacterList_unlink(&srcLine->CharacterList, srcCharacterReference);
}
void _WED_MoveCharacterToEndOfLine(
	WED_t *wed,
	_WED_Line_t *srcLine, WED_LineReference_t srcLineReference,
	WED_CharacterReference_t srcCharacterReference, _WED_Character_t *srcCharacter,
	_WED_Line_t *dstLine, WED_LineReference_t dstLineReference
){
	WED_CharacterReference_t dstCharacterReference = _WED_CharacterList_NewNode(&dstLine->CharacterList);
	_WED_CharacterList_linkPrev(&dstLine->CharacterList, dstLine->CharacterList.dst, dstCharacterReference);
	_WED_CharacterList_Node_t *dstCharacterNode = _WED_CharacterList_GetNodeByReference(
		&dstLine->CharacterList,
		dstCharacterReference
	);
	_WED_Character_t *dstCharacter = &dstCharacterNode->data.data;
	if(srcCharacter->CursorReference != -1){
		_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(
			&wed->CursorList,
			srcCharacter->CursorReference
		);
		WED_Cursor_t *Cursor = &CursorNode->data.data;
		Cursor->FreeStyle.CharacterReference = dstCharacterReference;
		Cursor->FreeStyle.LineReference = dstLineReference;
	}
	srcLine->TotalWidth -= srcCharacter->width;
	dstLine->TotalWidth += srcCharacter->width;
	dstCharacter->CursorReference = srcCharacter->CursorReference;
	dstCharacter->width = srcCharacter->width;
	dstCharacter->data = srcCharacter->data;
	_WED_CharacterList_unlink(&srcLine->CharacterList, srcCharacterReference);
}

void _WED_MoveCursorFreeStyle(
	WED_t *wed,
	WED_CursorReference_t CursorReference,
	WED_Cursor_t *Cursor,
	WED_LineReference_t dstLineReference,
	WED_CharacterReference_t dstCharacterReference,
	_WED_Character_t *dstCharacter
){
	_WED_LineList_Node_t *srcLineNode = _WED_LineList_GetNodeByReference(
		&wed->LineList,
		Cursor->FreeStyle.LineReference
	);
	_WED_Line_t *srcLine = &srcLineNode->data.data;
	_WED_CharacterList_Node_t *srcCharacterNode = _WED_CharacterList_GetNodeByReference(
		&srcLine->CharacterList,
		Cursor->FreeStyle.CharacterReference
	);
	_WED_Character_t *srcCharacter = &srcCharacterNode->data.data;
	srcCharacter->CursorReference = -1;
	if(dstCharacter->CursorReference != -1){
		/* there is already cursor what should we do? */
		assert(0);
	}
	dstCharacter->CursorReference = CursorReference;
	Cursor->FreeStyle.LineReference = dstLineReference;
	Cursor->FreeStyle.CharacterReference = dstCharacterReference;
}
void _WED_MoveCursor_NoCleaning(
	WED_CursorReference_t CursorReference,
	WED_LineReference_t *srcLineReference, /* will be changed */
	WED_CharacterReference_t *srcCharacterReference, /* will be changed */
	WED_LineReference_t dstLineReference,
	WED_CharacterReference_t dstCharacterReference,
	_WED_Character_t *dstCharacter
){
	if(dstCharacter->CursorReference != -1){
		/* there is already cursor what should we do? */
		assert(0);
	}
	dstCharacter->CursorReference = CursorReference;
	*srcLineReference = dstLineReference;
	*srcCharacterReference = dstCharacterReference;
}
void _WED_MoveCursorSelection(
	WED_CursorReference_t CursorReference,
	WED_LineReference_t *srcLineReference, /* will be changed */
	WED_CharacterReference_t *srcCharacterReference, /* will be changed */
	_WED_Character_t *srcCharacter,
	WED_LineReference_t dstLineReference,
	WED_CharacterReference_t dstCharacterReference,
	_WED_Character_t *dstCharacter
){
	srcCharacter->CursorReference = -1;
	_WED_MoveCursor_NoCleaning(
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
void _WED_MoveAllCursors(
	WED_t *wed,
	WED_LineReference_t srcLineReference,
	WED_CharacterReference_t srcCharacterReference,
	_WED_Character_t *srcCharacter,
	WED_LineReference_t dstLineReference,
	WED_CharacterReference_t dstCharacterReference,
	_WED_Character_t *dstCharacter
){
	WED_CursorReference_t CursorReference = srcCharacter->CursorReference;
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
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			Cursor->FreeStyle.LineReference = dstLineReference;
			Cursor->FreeStyle.CharacterReference = dstCharacterReference;
			break;
		}
		case WED_CursorType_Selection_e:{
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

void _WED_RemoveCharacter_Safe(
	WED_t *wed,
	WED_LineReference_t LineReference,
	_WED_Line_t *Line,
	WED_CharacterReference_t CharacterReference,
	_WED_Character_t *Character
){
	WED_CharacterReference_t dstCharacterReference = _WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		CharacterReference
	)->PrevNodeReference;
	_WED_CharacterList_Node_t *dstCharacterNode = _WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		dstCharacterReference
	);
	_WED_Character_t *dstCharacter = &dstCharacterNode->data.data;
	_WED_MoveAllCursors(wed, LineReference, CharacterReference, Character, LineReference, dstCharacterReference, dstCharacter);

	Line->TotalWidth -= Character->width;
	_WED_CharacterList_unlink(&Line->CharacterList, CharacterReference);
}
void _WED_RemoveCharacter_Unsafe(
	_WED_Line_t *Line,
	WED_CharacterReference_t CharacterReference,
	_WED_Character_t *Character
){
	Line->TotalWidth -= Character->width;
	_WED_CharacterList_unlink(&Line->CharacterList, CharacterReference);
}

/* returns 0 if left is possible */
bool _WED_GetLineAndCharacterOfLeft(
	WED_t *wed,
	WED_LineReference_t srcLineReference,
	_WED_LineList_Node_t *srcLineNode,
	WED_CharacterReference_t srcCharacterReference,
	WED_LineReference_t *dstLineReference,
	_WED_Line_t **dstLine,
	WED_CharacterReference_t *dstCharacterReference
){
	_WED_Line_t *srcLine = &srcLineNode->data.data;
	if(srcCharacterReference == srcLine->CharacterList.src){
		/* its begin of line. can we go up? */
		WED_LineReference_t PrevLineReference = srcLineNode->PrevNodeReference;
		if(PrevLineReference != wed->LineList.src){
			_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_WED_Line_t *PrevLine = &PrevLineNode->data.data;
			*dstLineReference = PrevLineReference;
			*dstLine = PrevLine;
			*dstCharacterReference = _WED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
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
		*dstCharacterReference = _WED_CharacterList_GetNodeByReference(
			&srcLine->CharacterList,
			srcCharacterReference
		)->PrevNodeReference;
		return 0;
	}
}
void _WED_GetLineAndCharacterOfLeft_Unsafe(
	WED_t *wed,
	WED_LineReference_t srcLineReference,
	_WED_LineList_Node_t *srcLineNode,
	WED_CharacterReference_t srcCharacterReference,
	WED_LineReference_t *dstLineReference,
	_WED_Line_t **dstLine,
	WED_CharacterReference_t *dstCharacterReference
){
	_WED_Line_t *srcLine = &srcLineNode->data.data;
	if(srcCharacterReference == srcLine->CharacterList.src){
		/* its begin of line. can we go up? */
		WED_LineReference_t PrevLineReference = srcLineNode->PrevNodeReference;
		_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
		_WED_Line_t *PrevLine = &PrevLineNode->data.data;
		*dstLineReference = PrevLineReference;
		*dstLine = PrevLine;
		*dstCharacterReference = _WED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
	}
	else{
		*dstLineReference = srcLineReference;
		*dstLine = srcLine;
		*dstCharacterReference = _WED_CharacterList_GetNodeByReference(
			&srcLine->CharacterList,
			srcCharacterReference
		)->PrevNodeReference;
	}
}
/* returns 0 if right is possible */
bool _WED_GetLineAndCharacterOfRight(
	WED_t *wed,
	WED_LineReference_t srcLineReference,
	_WED_LineList_Node_t *srcLineNode,
	WED_CharacterReference_t srcCharacterReference,
	WED_LineReference_t *dstLineReference,
	_WED_Line_t **dstLine,
	WED_CharacterReference_t *dstCharacterReference
){
	_WED_Line_t *srcLine = &srcLineNode->data.data;
	WED_CharacterReference_t NextCharacterReference = _WED_CharacterList_GetNodeByReference(
		&srcLine->CharacterList,
		srcCharacterReference
	)->NextNodeReference;
	if(NextCharacterReference == srcLine->CharacterList.dst){
		/* its end of line. can we go up? */
		WED_LineReference_t NextLineReference = srcLineNode->NextNodeReference;
		if(NextLineReference != wed->LineList.dst){
			_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_WED_Line_t *NextLine = &NextLineNode->data.data;
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
void _WED_GetLineAndCharacterOfRight_Unsafe(
	WED_t *wed,
	WED_LineReference_t srcLineReference,
	_WED_LineList_Node_t *srcLineNode,
	WED_CharacterReference_t srcCharacterReference,
	WED_LineReference_t *dstLineReference,
	_WED_Line_t **dstLine,
	WED_CharacterReference_t *dstCharacterReference
){
	_WED_Line_t *srcLine = &srcLineNode->data.data;
	WED_CharacterReference_t NextCharacterReference = _WED_CharacterList_GetNodeByReference(
		&srcLine->CharacterList,
		srcCharacterReference
	)->NextNodeReference;
	if(NextCharacterReference == srcLine->CharacterList.dst){
		/* its end of line */
		WED_LineReference_t NextLineReference = srcLineNode->NextNodeReference;
		_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_WED_Line_t *NextLine = &NextLineNode->data.data;
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
void _WED_OpenExtraLine(
	WED_t *wed,
	WED_LineReference_t LineReference,
	WED_LineReference_t *NextLineReference,
	_WED_LineList_Node_t **NextLineNode
){
	*NextLineReference = _WED_LineList_NewNode(&wed->LineList);
	*NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *NextLineReference);
	_WED_Line_t *NextLine = &(*NextLineNode)->data.data;
	_WED_LineList_linkNext(&wed->LineList, LineReference, *NextLineReference);
	NextLine->TotalWidth = 0;
	_WED_Line_t *Line = &_WED_LineList_GetNodeByReference(&wed->LineList, LineReference)->data.data;
	if(Line->IsEndLine){
		Line->IsEndLine = 0;
		NextLine->IsEndLine = 1;
	}
	else{
		NextLine->IsEndLine = 0;
	}
	_WED_CharacterList_open(&NextLine->CharacterList);
	_WED_CharacterList_Node_t *GodCharacterNode = _WED_CharacterList_GetNodeByReference(
		&NextLine->CharacterList,
		NextLine->CharacterList.src
	);
	_WED_Character_t *GodCharacter = &GodCharacterNode->data.data;
	GodCharacter->CursorReference = -1;
}

void _WED_IncreaseSlideToPrev(WED_t *wed, WED_LineReference_t LineReference, _WED_LineList_Node_t *LineNode){
	WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
	if(PrevLineReference == wed->LineList.src){
		/* we are first line */
		return;
	}
	_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
	_WED_Line_t *PrevLine = &PrevLineNode->data.data;
	if(PrevLine->IsEndLine == 1){
		/* previous line is not part of this line */
		return;
	}
	_WED_Line_t *Line = &LineNode->data.data;
	WED_CharacterReference_t FirstCharacterReference = _WED_CharacterList_GetNodeFirst(&Line->CharacterList);
	_WED_CharacterList_Node_t *FirstCharacterNode = _WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		FirstCharacterReference
	);
	_WED_Character_t *FirstCharacter = &FirstCharacterNode->data.data;
	if((FirstCharacter->width + PrevLine->TotalWidth) <= wed->LineWidth){
		_WED_MoveCharacterToEndOfLine(
			wed,
			Line,
			LineReference,
			FirstCharacterReference,
			FirstCharacter,
			PrevLine,
			PrevLineReference
		);
	}
}
void _WED_DecreaseSlideToPrev_Loop(
	WED_t *wed,
	WED_LineReference_t LineReference,
	_WED_LineList_Node_t *LineNode,
	WED_LineReference_t NextLineReference,
	_WED_LineList_Node_t *NextLineNode
){
	NewLineBegin:
	WED_CharacterReference_t CharacterReference = _WED_CharacterList_GetNodeFirst(
		&NextLineNode->data.data.CharacterList
	);
	_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(&NextLineNode->data.data.CharacterList, CharacterReference)->data.data;
	if((Character->width + LineNode->data.data.TotalWidth) > wed->LineWidth){
		LineNode = NextLineNode;
		LineReference = NextLineReference;
		if(NextLineNode->data.data.IsEndLine == 0){
			NextLineReference = LineNode->NextNodeReference;
			NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			goto NewLineBegin;
		}
		else{
			return;
		}
	}
	_WED_Character_t *GodCharacter = &_WED_CharacterList_GetNodeByReference(
		&NextLineNode->data.data.CharacterList,
		NextLineNode->data.data.CharacterList.src
	)->data.data;
	WED_CharacterReference_t LastCharacterReference = _WED_CharacterList_GetNodeLast(&LineNode->data.data.CharacterList);
	_WED_Character_t *LastCharacter = &_WED_CharacterList_GetNodeByReference(
		&LineNode->data.data.CharacterList,
		LastCharacterReference
	)->data.data;
	_WED_MoveAllCursors(
		wed,
		NextLineReference,
		NextLineNode->data.data.CharacterList.src,
		GodCharacter,
		LineReference,
		LastCharacterReference,
		LastCharacter
	);
	begin:
	if((Character->width + LineNode->data.data.TotalWidth) > wed->LineWidth){
		/* doesnt fit more */
		if(NextLineNode->data.data.IsEndLine == 0){
			LineReference = NextLineReference;
			LineNode = NextLineNode;
			NextLineReference = LineNode->NextNodeReference;
			NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			goto NewLineBegin;
		}
		else{
			/* we cant loop next line because it has endline */
			return;
		}
	}
	_WED_MoveCharacterToEndOfLine(
		wed,
		&NextLineNode->data.data,
		NextLineReference,
		CharacterReference,
		Character,
		&LineNode->data.data,
		LineReference
	);
	CharacterReference = _WED_CharacterList_GetNodeFirst(&NextLineNode->data.data.CharacterList);
	if(CharacterReference == NextLineNode->data.data.CharacterList.dst){
		/* so lets delete this line */
		if(NextLineNode->data.data.IsEndLine == 1){
			/* we will dont loop more */
			LineNode->data.data.IsEndLine = 1;
			_WED_LineList_unlink(&wed->LineList, NextLineReference);
			return;
		}
		else{
			/* we will loop more */
			_WED_LineList_unlink(&wed->LineList, NextLineReference);
			NextLineReference = LineNode->NextNodeReference;
			NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			goto NewLineBegin;
		}
	}
	Character = &_WED_CharacterList_GetNodeByReference(
		&NextLineNode->data.data.CharacterList,
		CharacterReference
	)->data.data;
	goto begin;
}
void _WED_DecreaseSlideToPrev_CheckNext(WED_t *wed, WED_LineReference_t LineReference, _WED_LineList_Node_t *LineNode){
	_WED_Line_t *Line = &LineNode->data.data;
	if(Line->IsEndLine == 1){
		/* next line cant be related with this line */
		return;
	}
	WED_LineReference_t NextLineReference = LineNode->NextNodeReference;
	if(NextLineReference == wed->LineList.dst){
		/* we are also bottom line */
		return;
	}
	_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
	_WED_DecreaseSlideToPrev_Loop(wed, LineReference, LineNode, NextLineReference, NextLineNode);
}
void _WED_DecreaseSlideToPrev(WED_t *wed, WED_LineReference_t LineReference, _WED_LineList_Node_t *LineNode){
	WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
	if(PrevLineReference == wed->LineList.src){
		/* we are first line */
		_WED_DecreaseSlideToPrev_CheckNext(wed, LineReference, LineNode);
		return;
	}
	_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
	_WED_Line_t *PrevLine = &PrevLineNode->data.data;
	if(PrevLine->IsEndLine == 1){
		/* previous line is not part of this line */
		_WED_DecreaseSlideToPrev_CheckNext(wed, LineReference, LineNode);
		return;
	}
	_WED_DecreaseSlideToPrev_Loop(wed, PrevLineReference, PrevLineNode, LineReference, LineNode);
}
void _WED_SlideToNext(WED_t *wed, WED_LineReference_t LineReference, _WED_LineList_Node_t *LineNode){
	begin:
	WED_LineReference_t NextLineReference;
	_WED_LineList_Node_t *NextLineNode;
	bool IsLoopEntered = LineNode->data.data.TotalWidth > wed->LineWidth;
	while(LineNode->data.data.TotalWidth > wed->LineWidth){
		if(LineNode->data.data.IsEndLine){
			/* if line has endline we need to create new line to slide */
			_WED_OpenExtraLine(wed, LineReference, &NextLineReference, &NextLineNode);
			/* that function maybe changed line address so lets get it again */
			LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		}
		else{
			NextLineReference = LineNode->NextNodeReference;
			NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		}
		WED_CharacterReference_t CharacterReference = _WED_CharacterList_GetNodeLast(&LineNode->data.data.CharacterList);
		_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
			&LineNode->data.data.CharacterList,
			CharacterReference
		)->data.data;
		_WED_MoveCharacterToBeginOfLine(
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

void _WED_LineIsIncreased(WED_t *wed, WED_LineReference_t LineReference, _WED_LineList_Node_t *LineNode){
	_WED_IncreaseSlideToPrev(wed, LineReference, LineNode);
	_WED_SlideToNext(wed, LineReference, LineNode);
}

void _WED_LineIsDecreased(WED_t *wed, WED_LineReference_t LineReference, _WED_LineList_Node_t *LineNode){
	_WED_Line_t *Line = &LineNode->data.data;
	if(_WED_CharacterList_usage(&Line->CharacterList) == 0){
		if(Line->IsEndLine == 1){
			WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference != wed->LineList.src){
				_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(
					&wed->LineList,
					PrevLineReference
				);
				_WED_Line_t *PrevLine = &PrevLineNode->data.data;
				if(PrevLine->IsEndLine == 0){
					PrevLine->IsEndLine = 1;
					_WED_Character_t *GodCharacter = &_WED_CharacterList_GetNodeByReference(
						&Line->CharacterList,
						Line->CharacterList.src
					)->data.data;
					WED_CharacterReference_t LastCharacterReference = _WED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
					_WED_Character_t *LastCharacter = &_WED_CharacterList_GetNodeByReference(
						&PrevLine->CharacterList,
						LastCharacterReference
					)->data.data;
					_WED_MoveAllCursors(
						wed,
						LineReference,
						Line->CharacterList.src,
						GodCharacter,
						PrevLineReference,
						LastCharacterReference,
						LastCharacter
					);
					_WED_LineList_unlink(&wed->LineList, LineReference);
					return;
				}
				else{
					/* previous line is not part of current line */
					return;
				}
			}
			else{
				/* we dont have previous line */
				return;
			}
		}
		else{
			_WED_Character_t *GodCharacter = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Line->CharacterList.src
			)->data.data;
			WED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_WED_Line_t *NextLine = &NextLineNode->data.data;
			WED_CharacterReference_t FirstCharacterReference = NextLine->CharacterList.src;
			_WED_Character_t *FirstCharacter = &_WED_CharacterList_GetNodeByReference(
				&NextLine->CharacterList,
				FirstCharacterReference
			)->data.data;
			_WED_MoveAllCursors(
				wed,
				LineReference,
				Line->CharacterList.src,
				GodCharacter,
				NextLineReference,
				FirstCharacterReference,
				FirstCharacter
			);
			_WED_LineList_unlink(&wed->LineList, LineReference);
			return;
		}
	}
	_WED_DecreaseSlideToPrev(wed, LineReference, LineNode);
	_WED_SlideToNext(wed, LineReference, LineNode);
}

void _WED_CursorIsTriggered(WED_Cursor_t *Cursor){
	/* this function must be called when something is changed or could change */
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			Cursor->FreeStyle.PreferredWidth = -1;
			break;
		}
		case WED_CursorType_Selection_e:{
			Cursor->Selection.PreferredWidth = -1;
			break;
		}
	}
}

void _WED_GetCharacterFromLineByWidth(
	WED_t *wed,
	_WED_Line_t *Line,
	uint32_t Width,
	WED_CharacterReference_t *pCharacterReference,
	_WED_Character_t **pCharacter
){
	uint32_t iWidth = 0;
	WED_CharacterReference_t CharacterReference = Line->CharacterList.src;
	while(1){
		WED_CharacterReference_t NextCharacterReference = _WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->NextNodeReference;
		if(NextCharacterReference == Line->CharacterList.dst){
			/* lets return Character */
			*pCharacter = &_WED_CharacterList_GetNodeByReference(&Line->CharacterList, CharacterReference)->data.data;
			*pCharacterReference = CharacterReference;
			return;
		}
		else{
			_WED_Character_t *NextCharacter = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				NextCharacterReference
			)->data.data;
			if((iWidth + NextCharacter->width) >= Width){
				/* we need to return Character or NextCharacter depends about how close to Width */
				uint32_t CurrentDiff = Width - iWidth;
				uint32_t NextDiff = iWidth + NextCharacter->width - Width;
				if(CurrentDiff <= NextDiff){
					/* lets return Character */
					*pCharacter = &_WED_CharacterList_GetNodeByReference(
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

uint32_t _WED_CalculatePositionOfCharacterInLine(WED_t *wed, _WED_Line_t *Line, WED_CharacterReference_t pCharacterReference){
	uint32_t Width = 0;
	WED_CharacterReference_t CharacterReference = _WED_CharacterList_GetNodeFirst(&Line->CharacterList);
	while(1){
		_WED_CharacterList_Node_t *CharacterNode = _WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		);
		_WED_Character_t *Character = &CharacterNode->data.data;
		Width += Character->width;
		if(CharacterReference == pCharacterReference){
			break;
		}
		CharacterReference = CharacterNode->NextNodeReference;
	}
	return Width;
}

void _WED_CalculatePreferredWidthIfNeeded(
	WED_t *wed,
	_WED_Line_t *Line,
	WED_CharacterReference_t CharacterReference,
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
		*PreferredWidth = _WED_CalculatePositionOfCharacterInLine(
			wed,
			Line,
			CharacterReference
		);
	}
}

void _WED_CursorConvertFreeStyleToSelection(
	WED_CursorReference_t CursorReference,
	WED_Cursor_t *Cursor,
	WED_LineReference_t LineReference,
	_WED_Line_t *Line,
	WED_CharacterReference_t CharacterReference,
	_WED_Character_t *Character,
	uint32_t PreferredWidth
){
	Cursor->type = WED_CursorType_Selection_e;
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

void _WED_CursorConvertSelectionToFreeStyle(WED_t *wed, WED_Cursor_t *Cursor, bool Direction){
	Cursor->type = WED_CursorType_FreeStyle_e;
	{
		WED_LineReference_t LineReference = Cursor->Selection.LineReference[Direction ^ 1];
		_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_WED_Line_t *Line = &LineNode->data.data;
		WED_CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction ^ 1];
		_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->data.data;
		Character->CursorReference = -1;
	}
	WED_LineReference_t LineReference = Cursor->Selection.LineReference[Direction];
	WED_CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction];
	uint32_t PreferredWidth = Cursor->Selection.PreferredWidth;
	Cursor->FreeStyle.LineReference = LineReference;
	Cursor->FreeStyle.CharacterReference = CharacterReference;
	Cursor->FreeStyle.PreferredWidth = PreferredWidth;
}

void _WED_CursorDeleteSelectedAndMakeCursorFreeStyle(WED_t *wed, WED_Cursor_t *Cursor){
	bool Direction;
	WED_LineReference_t LineReference0;
	WED_LineReference_t LineReference;
	if(Cursor->Selection.LineReference[0] != Cursor->Selection.LineReference[1]){
		/* lets compare lines */
		if(_WED_LineList_IsNodeReferenceFronter(
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
		_WED_Line_t *Line = &_WED_LineList_GetNodeByReference(
			&wed->LineList,
			Cursor->Selection.LineReference[0]
		)->data.data;
		if(_WED_CharacterList_IsNodeReferenceFronter(
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
	WED_CharacterReference_t CharacterReference0 = Cursor->Selection.CharacterReference[Direction];
	WED_CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction ^ 1];
	_WED_LineList_Node_t *LineNode0 = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference0);
	_WED_Line_t *Line0 = &LineNode0->data.data;
	_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, Direction);
	while(1){
		_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_WED_Line_t *Line = &LineNode->data.data;
		_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->data.data;
		_WED_Character_t *Character0 = &_WED_CharacterList_GetNodeByReference(
			&Line0->CharacterList,
			CharacterReference0
		)->data.data;
		_WED_MoveAllCursors(
			wed,
			LineReference,
			CharacterReference,
			Character,
			LineReference0,
			CharacterReference0,
			Character0
		);
		WED_LineReference_t dstLineReference;
		_WED_Line_t *dstLine;
		WED_CharacterReference_t dstCharacterReference;
		if(CharacterReference != Line->CharacterList.src){
			_WED_RemoveCharacter_Unsafe(Line, CharacterReference, Character);
			_WED_GetLineAndCharacterOfLeft_Unsafe(
				wed,
				LineReference,
				LineNode,
				CharacterReference,
				&dstLineReference,
				&dstLine,
				&dstCharacterReference
			);
			if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
				/* we did reach where we go */
				_WED_LineIsDecreased(wed, LineReference, LineNode);
				return;
			}
		}
		else{
			/* we need to delete something from previous line */
			BeginOfCharacterReferenceIsFirst:
			dstLineReference = LineNode->PrevNodeReference;
			_WED_LineList_Node_t *dstLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, dstLineReference);
			dstLine = &dstLineNode->data.data;
			dstCharacterReference = _WED_CharacterList_GetNodeLast(&dstLine->CharacterList);
			if(dstLine->IsEndLine == 1){
				if(_WED_CharacterList_usage(&dstLine->CharacterList) == 0){
					/* lets delete line */
					_WED_Character_t *srcGodCharacter = &_WED_CharacterList_GetNodeByReference(
						&Line->CharacterList,
						Line->CharacterList.src
					)->data.data;
					_WED_Character_t *dstGodCharacter = &_WED_CharacterList_GetNodeByReference(
						&dstLine->CharacterList,
						dstLine->CharacterList.src
					)->data.data;
					_WED_MoveAllCursors(
						wed,
						dstLineReference,
						dstCharacterReference,
						dstGodCharacter,
						LineReference,
						CharacterReference,
						srcGodCharacter
					);
					_WED_LineList_unlink(&wed->LineList, dstLineReference);
					if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
						/* we did reach where we go */
						_WED_LineIsDecreased(wed, LineReference, LineNode);
						return;
					}
					goto BeginOfCharacterReferenceIsFirst;
				}
				else{
					dstLine->IsEndLine = 0;
					_WED_LineIsDecreased(wed, dstLineReference, dstLineNode);
					if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
						/* we did reach where we go */
						_WED_LineIsDecreased(wed, LineReference, LineNode);
						return;
					}
				}
			}
			else{
				/* nothing to delete */
				if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
					/* we did reach where we go */
					_WED_LineIsDecreased(wed, LineReference, LineNode);
					return;
				}
			}
		}
		if(LineReference != dstLineReference){
			/* we got other line */
			_WED_LineIsDecreased(wed, LineReference, LineNode);
			LineReference = dstLineReference;
		}
		CharacterReference = dstCharacterReference;
	}
}
