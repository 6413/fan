static
void FED_open(FED_t *wed, uint32_t LineHeight, uint32_t LineWidth, uint32_t LineLimit, uint32_t LineCharacterLimit){
	_FED_LineList_open(&wed->LineList);
	_FED_CursorList_open(&wed->CursorList);
	wed->LineHeight = LineHeight;
	wed->LineWidth = LineWidth;
	wed->LineLimit = LineLimit;
	wed->LineCharacterLimit = LineCharacterLimit;
}
static
void FED_close(FED_t *wed){
	_FED_CursorList_close(&wed->CursorList);
	FED_LineReference_t LineReference = _FED_LineList_GetNodeFirst(&wed->LineList);
	while(LineReference != wed->LineList.dst){
		_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_FED_Line_t *Line = &LineNode->data.data;
		_FED_CharacterList_close(&Line->CharacterList);
		LineReference = LineNode->NextNodeReference;
	}
	_FED_LineList_close(&wed->LineList);
}

/* O(LineNumber) */
static
FED_LineReference_t FED_GetLineReferenceByLineIndex(FED_t *wed, uint32_t LineNumber){
	if(LineNumber >= _FED_LineList_usage(&wed->LineList)){
		return -1;
	}
	FED_LineReference_t LineReference = _FED_LineList_GetNodeByReference(
		&wed->LineList,
		wed->LineList.src
	)->NextNodeReference;
	for(; LineNumber; LineNumber--){
		LineReference = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference)->NextNodeReference;
	}
	return LineReference;
}

static
void FED_EndLine(FED_t *wed, FED_CursorReference_t CursorReference){
	if(_FED_LineList_usage(&wed->LineList) == wed->LineLimit){
		return;
	}
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
	FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
	FED_LineReference_t NextLineReference = _FED_LineList_NewNode(&wed->LineList);
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	if(Line->IsEndLine == 0){
		Line->IsEndLine = 1;
		_FED_LineList_linkNext(&wed->LineList, LineReference, NextLineReference);
		_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_FED_Line_t *NextLine = &NextLineNode->data.data;
		NextLine->TotalWidth = 0;
		NextLine->IsEndLine = 0;
		_FED_CharacterList_open(&NextLine->CharacterList);
		_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_FED_Line_t *Line = &LineNode->data.data;
		FED_CharacterReference_t CharacterReference = _FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Cursor->FreeStyle.CharacterReference
		)->NextNodeReference;
		while(CharacterReference != Line->CharacterList.dst){
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_FED_MoveCharacterToEndOfLine(
				wed,
				Line,
				LineReference,
				CharacterReference,
				Character,
				NextLine,
				NextLineReference
			);
			CharacterReference = _FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->NextNodeReference;
		}
		_FED_Character_t *NextLineFirstCharacter = &_FED_CharacterList_GetNodeByReference(
			&NextLine->CharacterList,
			NextLine->CharacterList.src
		)->data.data;
		NextLineFirstCharacter->CursorReference = -1;
		_FED_MoveCursorFreeStyle(
			wed,
			CursorReference,
			Cursor,
			NextLineReference,
			NextLine->CharacterList.src,
			NextLineFirstCharacter
		);
		_FED_LineIsDecreased(wed, NextLineReference, NextLineNode);
	}
	else{
		_FED_LineList_linkNext(&wed->LineList, LineReference, NextLineReference);
		_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_FED_Line_t *NextLine = &NextLineNode->data.data;
		NextLine->TotalWidth = 0;
		NextLine->IsEndLine = 1;
		_FED_CharacterList_open(&NextLine->CharacterList);
		_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_FED_Line_t *Line = &LineNode->data.data;
		FED_CharacterReference_t CharacterReference = _FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Cursor->FreeStyle.CharacterReference
		)->NextNodeReference;
		while(CharacterReference != Line->CharacterList.dst){
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_FED_MoveCharacterToEndOfLine(
				wed,
				Line,
				LineReference,
				CharacterReference,
				Character,
				NextLine,
				NextLineReference
			);
			CharacterReference = _FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->NextNodeReference;
		}
		_FED_Character_t *NextLineFirstCharacter = &_FED_CharacterList_GetNodeByReference(
			&NextLine->CharacterList,
			NextLine->CharacterList.src
		)->data.data;
		NextLineFirstCharacter->CursorReference = -1;
		_FED_MoveCursorFreeStyle(
			wed,
			CursorReference,
			Cursor,
			NextLineReference,
			NextLine->CharacterList.src,
			NextLineFirstCharacter
		);
	}
}

static
void FED_MoveCursorFreeStyleToLeft(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			if(_FED_GetLineAndCharacterOfLeft(
				wed,
				LineReference,
				LineNode,
				CharacterReference,
				&LineReference,
				&Line,
				&CharacterReference)
			){
				return;
			}
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_FED_MoveCursorFreeStyle(wed, CursorReference, Cursor, LineReference, CharacterReference, Character);
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
}
static
void FED_MoveCursorFreeStyleToRight(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line;
			FED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			if(_FED_GetLineAndCharacterOfRight(
				wed,
				LineReference,
				LineNode,
				CharacterReference,
				&LineReference,
				&Line,
				&CharacterReference)
			){
				return;
			}
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_FED_MoveCursorFreeStyle(wed, CursorReference, Cursor, LineReference, CharacterReference, Character);
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
			break;
		}
	}
}

static
void FED_MoveCursorSelectionToLeft(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			FED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			FED_LineReference_t LeftLineReference;
			_FED_Line_t *LeftLine;
			if(_FED_GetLineAndCharacterOfLeft(
				wed,
				LineReference,
				LineNode,
				CharacterReference,
				&LeftLineReference,
				&LeftLine,
				&CharacterReference
			)){
				return;
			}
			_FED_Character_t *LeftCharacter = &_FED_CharacterList_GetNodeByReference(
				&LeftLine->CharacterList,
				CharacterReference
			)->data.data;
			_FED_CursorConvertFreeStyleToSelection(
				CursorReference,
				Cursor,
				LeftLineReference,
				LeftLine,
				CharacterReference,
				LeftCharacter,
				-1
			);
			break;
		}
		case FED_CursorType_Selection_e:{
			FED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			FED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_LineReference_t LeftLineReference;
			_FED_Line_t *LeftLine;
			FED_CharacterReference_t LeftCharacterReference;
			if(_FED_GetLineAndCharacterOfLeft(
				wed,
				*LineReference,
				LineNode,
				*CharacterReference,
				&LeftLineReference,
				&LeftLine,
				&LeftCharacterReference
			)){
				return;
			}
			if(
				Cursor->Selection.CharacterReference[0] == LeftCharacterReference &&
				Cursor->Selection.LineReference[0] == LeftLineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_FED_Character_t *LeftCharacter = &_FED_CharacterList_GetNodeByReference(
				&LeftLine->CharacterList,
				LeftCharacterReference
			)->data.data;
			_FED_MoveCursorSelection(
				CursorReference,
				LineReference,
				CharacterReference,
				Character,
				LeftLineReference,
				LeftCharacterReference,
				LeftCharacter
			);
			break;
		}
	}
}
static
void FED_MoveCursorSelectionToRight(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			FED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			FED_LineReference_t RightLineReference;
			_FED_Line_t *RightLine;
			if(_FED_GetLineAndCharacterOfRight(
				wed,
				LineReference,
				LineNode,
				CharacterReference,
				&RightLineReference,
				&RightLine,
				&CharacterReference
			)){
				return;
			}
			_FED_Character_t *RightCharacter = &_FED_CharacterList_GetNodeByReference(
				&RightLine->CharacterList,
				CharacterReference
			)->data.data;
			_FED_CursorConvertFreeStyleToSelection(
				CursorReference,
				Cursor,
				RightLineReference,
				RightLine,
				CharacterReference,
				RightCharacter,
				-1
			);
			break;
		}
		case FED_CursorType_Selection_e:{
			FED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			FED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_LineReference_t RightLineReference;
			_FED_Line_t *RightLine;
			FED_CharacterReference_t RightCharacterReference;
			if(_FED_GetLineAndCharacterOfRight(
				wed,
				*LineReference,
				LineNode,
				*CharacterReference,
				&RightLineReference,
				&RightLine,
				&RightCharacterReference
			)){
				return;
			}
			if(
				Cursor->Selection.CharacterReference[0] == RightCharacterReference &&
				Cursor->Selection.LineReference[0] == RightLineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_FED_Character_t *RightCharacter = &_FED_CharacterList_GetNodeByReference(
				&RightLine->CharacterList,
				RightCharacterReference
			)->data.data;
			_FED_MoveCursorSelection(
				CursorReference,
				LineReference,
				CharacterReference,
				Character,
				RightLineReference,
				RightCharacterReference,
				RightCharacter
			);
			break;
		}
	}
}

static
void FED_MoveCursorFreeStyleToUp(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			FED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference == wed->LineList.src){
				/* we already in top */
				return;
			}
			_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_FED_Line_t *PrevLine = &PrevLineNode->data.data;
			FED_CharacterReference_t dstCharacterReference;
			_FED_Character_t *dstCharacter;
			_FED_GetCharacterFromLineByWidth(
				wed,
				PrevLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_FED_MoveCursorFreeStyle(wed, CursorReference, Cursor, PrevLineReference, dstCharacterReference, dstCharacter);
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
}
static
void FED_MoveCursorFreeStyleToDown(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			FED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			if(NextLineReference == wed->LineList.dst){
				/* we already in bottom */
				return;
			}
			_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_FED_Line_t *NextLine = &NextLineNode->data.data;
			FED_CharacterReference_t dstCharacterReference;
			_FED_Character_t *dstCharacter;
			_FED_GetCharacterFromLineByWidth(
				wed,
				NextLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_FED_MoveCursorFreeStyle(wed, CursorReference, Cursor, NextLineReference, dstCharacterReference, dstCharacter);
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
			break;
		}
	}
}

static
void FED_MoveCursorSelectionToUp(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			FED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference == wed->LineList.src){
				/* we already in top */
				return;
			}
			_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_FED_Line_t *PrevLine = &PrevLineNode->data.data;
			FED_CharacterReference_t dstCharacterReference;
			_FED_Character_t *dstCharacter;
			_FED_GetCharacterFromLineByWidth(
				wed,
				PrevLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_FED_CursorConvertFreeStyleToSelection(
				CursorReference,
				Cursor,
				PrevLineReference,
				PrevLine,
				dstCharacterReference,
				dstCharacter,
				Cursor->FreeStyle.PreferredWidth
			);
			break;
		}
		case FED_CursorType_Selection_e:{
			FED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			FED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				*CharacterReference,
				&Cursor->Selection.PreferredWidth
			);
			FED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference == wed->LineList.src){
				/* we already in top */
				return;
			}
			_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_FED_Line_t *PrevLine = &PrevLineNode->data.data;
			FED_CharacterReference_t dstCharacterReference;
			_FED_Character_t *dstCharacter;
			_FED_GetCharacterFromLineByWidth(
				wed,
				PrevLine,
				Cursor->Selection.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			if(
				Cursor->Selection.CharacterReference[0] == dstCharacterReference &&
				Cursor->Selection.LineReference[0] == PrevLineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_FED_MoveCursorSelection(
				CursorReference,
				LineReference,
				CharacterReference,
				Character,
				PrevLineReference,
				dstCharacterReference,
				dstCharacter
			);
			break;
		}
	}
}
static
void FED_MoveCursorSelectionToDown(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			FED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			if(NextLineReference == wed->LineList.dst){
				/* we already in bottom */
				return;
			}
			_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_FED_Line_t *NextLine = &NextLineNode->data.data;
			FED_CharacterReference_t dstCharacterReference;
			_FED_Character_t *dstCharacter;
			_FED_GetCharacterFromLineByWidth(
				wed,
				NextLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_FED_CursorConvertFreeStyleToSelection(
				CursorReference,
				Cursor,
				NextLineReference,
				NextLine,
				dstCharacterReference,
				dstCharacter,
				Cursor->FreeStyle.PreferredWidth
			);
			break;
		}
		case FED_CursorType_Selection_e:{
			FED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			FED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				*CharacterReference,
				&Cursor->Selection.PreferredWidth
			);
			FED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			if(NextLineReference == wed->LineList.dst){
				/* we already in bottom */
				return;
			}
			_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_FED_Line_t *NextLine = &NextLineNode->data.data;
			FED_CharacterReference_t dstCharacterReference;
			_FED_Character_t *dstCharacter;
			_FED_GetCharacterFromLineByWidth(
				wed,
				NextLine,
				Cursor->Selection.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			if(
				Cursor->Selection.CharacterReference[0] == dstCharacterReference &&
				Cursor->Selection.LineReference[0] == NextLineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_FED_MoveCursorSelection(
				CursorReference,
				LineReference,
				CharacterReference,
				Character,
				NextLineReference,
				dstCharacterReference,
				dstCharacter
			);
			break;
		}
	}
}

static
void FED_AddCharacterToCursor(FED_t *wed, FED_CursorReference_t CursorReference, FED_Data_t data, uint16_t width){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
	FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_FED_Line_t *Line = &LineNode->data.data;

	if (_FED_CharacterList_usage(&Line->CharacterList) == wed->LineCharacterLimit) {
		return;
	}
	if (Line->TotalWidth + width > wed->LineWidth) {
		return;
	}

	FED_CharacterReference_t CharacterReference = _FED_CharacterList_NewNode(&Line->CharacterList);
	_FED_CharacterList_linkNext(&Line->CharacterList, Cursor->FreeStyle.CharacterReference, CharacterReference);
	_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		CharacterReference
	)->data.data;
	Character->CursorReference = -1;
	Character->width = width;
	Character->data = data;
	Line->TotalWidth += width;
	_FED_MoveCursorFreeStyle(wed, CursorReference, Cursor, LineReference, CharacterReference, Character);
	_FED_LineIsIncreased(wed, LineReference, LineNode);
}

static
void FED_DeleteCharacterFromCursor(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			if(CharacterReference == Line->CharacterList.src){
				/* nothing to delete but can we delete something from previous line? */
				FED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
				if(PrevLineReference != wed->LineList.src){
					_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(
						&wed->LineList,
						PrevLineReference
					);
					_FED_Line_t *PrevLine = &PrevLineNode->data.data;
					if(PrevLine->IsEndLine){
						_FED_Character_t *GodCharacter = &_FED_CharacterList_GetNodeByReference(
							&Line->CharacterList,
							Line->CharacterList.src
						)->data.data;
						FED_CharacterReference_t PrevLineLastCharacterReference = _FED_CharacterList_GetNodeLast(
							&PrevLine->CharacterList
						);
						_FED_Character_t *PrevLineLastCharacter = &_FED_CharacterList_GetNodeByReference(
							&PrevLine->CharacterList,
							PrevLineLastCharacterReference
						)->data.data;
						_FED_MoveAllCursors(
							wed,
							LineReference,
							Line->CharacterList.src,
							GodCharacter,
							PrevLineReference,
							PrevLineLastCharacterReference,
							PrevLineLastCharacter
						);
						PrevLine->IsEndLine = 0;
						_FED_LineIsDecreased(wed, PrevLineReference, PrevLineNode);
						return;
					}
					else{
						CharacterReference = _FED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
						_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
							&PrevLine->CharacterList,
							CharacterReference
						)->data.data;
						_FED_RemoveCharacter_Safe(
							wed,
							PrevLineReference,
							PrevLine,
							CharacterReference,
							Character
						);
						_FED_LineIsDecreased(wed, PrevLineReference, PrevLineNode);
					}
				}
				else{
					/* previous line doesnt exists at all */
					return;
				}
			}
			else{
				_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
					&Line->CharacterList,
					CharacterReference
				)->data.data;
				_FED_RemoveCharacter_Safe(wed, LineReference, Line, CharacterReference, Character);
				_FED_LineIsDecreased(wed, LineReference, LineNode);
			}
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
}

static
void FED_DeleteCharacterFromCursorRight(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_CharacterReference_t CharacterReference = _FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->NextNodeReference;
			if(CharacterReference == Line->CharacterList.dst){
				/* we are in end of line */
				FED_LineReference_t NextLineReference = LineNode->NextNodeReference;
				if(Line->IsEndLine == 1){
					/* lets delete endline */
					if(NextLineReference != wed->LineList.dst){
						_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(
							&wed->LineList,
							NextLineReference
						);
						_FED_Line_t *NextLine = &NextLineNode->data.data;
						if(_FED_CharacterList_usage(&NextLine->CharacterList)){
							Line->IsEndLine = 0;
							_FED_LineIsDecreased(wed, LineReference, LineNode);
						}
						else{
							/* next line doesnt have anything so lets just unlink it */
							_FED_Character_t *NextLineGodCharacter = &_FED_CharacterList_GetNodeByReference(
								&NextLine->CharacterList,
								NextLine->CharacterList.src
							)->data.data;
							_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
								&Line->CharacterList,
								CharacterReference
							)->data.data;
							_FED_MoveAllCursors(
								wed,
								NextLineReference,
								NextLine->CharacterList.src,
								NextLineGodCharacter,
								LineReference,
								CharacterReference,
								Character
							);
							_FED_RemoveLine(wed, NextLineReference);
						}
						return;
					}
					else{
						/* this is last line so we cant delete it */
						return;
					}
				}
				else{
					/* lets get nextline and delete first character of it */
					_FED_LineList_Node_t *NextLineNode = _FED_LineList_GetNodeByReference(
						&wed->LineList,
						NextLineReference
					);
					_FED_Line_t *NextLine = &NextLineNode->data.data;
					CharacterReference = _FED_CharacterList_GetNodeFirst(&NextLine->CharacterList);
					_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
						&Line->CharacterList,
						CharacterReference
					)->data.data;
					_FED_RemoveCharacter_Safe(wed, LineReference, Line, CharacterReference, Character);
					_FED_LineIsDecreased(wed, NextLineReference, NextLineNode);
				}
			}
			else{
				_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
					&Line->CharacterList,
					CharacterReference
				)->data.data;
				_FED_RemoveCharacter_Safe(wed, LineReference, Line, CharacterReference, Character);
				_FED_LineIsDecreased(wed, LineReference, LineNode);
			}
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
}

static
void FED_MoveCursorFreeStyleToBeginOfLine(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, Cursor->FreeStyle.LineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		Cursor->FreeStyle.CharacterReference
	)->data.data;
	Character->CursorReference = -1;
	FED_CharacterReference_t BeginCharacterReference = Line->CharacterList.src;
	_FED_Character_t *BeginCharacter = &_FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		BeginCharacterReference
	)->data.data;
	if(BeginCharacter->CursorReference != -1){
		/* there is already cursor there */
		assert(0);
	}
	BeginCharacter->CursorReference = CursorReference;
	Cursor->FreeStyle.CharacterReference = BeginCharacterReference;
}
static
void FED_MoveCursorFreeStyleToEndOfLine(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, Cursor->FreeStyle.LineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		Cursor->FreeStyle.CharacterReference
	)->data.data;
	Character->CursorReference = -1;
	FED_CharacterReference_t EndCharacterReference = _FED_CharacterList_GetNodeLast(&Line->CharacterList);
	_FED_Character_t *EndCharacter = &_FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		EndCharacterReference
	)->data.data;
	if(EndCharacter->CursorReference != -1){
		/* there is already cursor there */
		assert(0);
	}
	EndCharacter->CursorReference = CursorReference;
	Cursor->FreeStyle.CharacterReference = EndCharacterReference;
}

static
void FED_MoveCursorSelectionToBeginOfLine(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_CharacterReference_t FirstCharacterReference = Line->CharacterList.src;
			_FED_Character_t *FirstCharacter = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				FirstCharacterReference
			)->data.data;
			_FED_CursorConvertFreeStyleToSelection(
				CursorReference,
				Cursor,
				LineReference,
				Line,
				FirstCharacterReference,
				FirstCharacter,
				-1
			);
			break;
		}
		case FED_CursorType_Selection_e:{
			FED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			FED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			FED_CharacterReference_t FirstCharacterReference = Line->CharacterList.src;
			if(
				Cursor->Selection.CharacterReference[0] == FirstCharacterReference &&
				Cursor->Selection.LineReference[0] == *LineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_FED_Character_t *FirstCharacter = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				FirstCharacterReference
			)->data.data;
			_FED_MoveCursorSelection(
				CursorReference,
				LineReference,
				CharacterReference,
				Character,
				*LineReference,
				FirstCharacterReference,
				FirstCharacter
			);
			break;
		}
	}
}
static
void FED_MoveCursorSelectionToEndOfLine(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			FED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			FED_CharacterReference_t LastCharacterReference = _FED_CharacterList_GetNodeLast(&Line->CharacterList);
			_FED_Character_t *LastCharacter = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				LastCharacterReference
			)->data.data;
			_FED_CursorConvertFreeStyleToSelection(
				CursorReference,
				Cursor,
				LineReference,
				Line,
				LastCharacterReference,
				LastCharacter,
				-1
			);
			break;
		}
		case FED_CursorType_Selection_e:{
			FED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			FED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			FED_CharacterReference_t LastCharacterReference = _FED_CharacterList_GetNodeLast(&Line->CharacterList);
			if(
				Cursor->Selection.CharacterReference[0] == LastCharacterReference &&
				Cursor->Selection.LineReference[0] == *LineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_FED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_FED_Character_t *LastCharacter = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				LastCharacterReference
			)->data.data;
			_FED_MoveCursorSelection(
				CursorReference,
				LineReference,
				CharacterReference,
				Character,
				*LineReference,
				LastCharacterReference,
				LastCharacter
			);
			break;
		}
	}
}

static
void FED_GetLineAndCharacter(
	FED_t *wed,
	FED_LineReference_t HintLineReference,
	uint32_t y,
	uint32_t x,
	FED_LineReference_t *LineReference, /* w */
	FED_CharacterReference_t *CharacterReference /* w */
){
	y /= wed->LineHeight;
	while(y--){
		HintLineReference = _FED_LineList_GetNodeByReference(&wed->LineList, HintLineReference)->NextNodeReference;
		if(HintLineReference == wed->LineList.dst){
			HintLineReference = _FED_LineList_GetNodeByReference(&wed->LineList, HintLineReference)->PrevNodeReference;
			x = 0xffffffff;
			break;
		}
	}
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, HintLineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	_FED_Character_t *unused;
	_FED_GetCharacterFromLineByWidth(wed, Line, x, CharacterReference, &unused);
	*LineReference = HintLineReference;
}

static
void _FED_UnlinkCursorFromCharacters(FED_t *wed, FED_CursorReference_t CursorReference, FED_Cursor_t *Cursor){
	switch(Cursor->type){
		case FED_CursorType_FreeStyle_e:{
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(
				&wed->LineList,
				Cursor->FreeStyle.LineReference
			);
			_FED_Line_t *Line = &LineNode->data.data;
			_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->data.data;
			Character->CursorReference = -1;
			break;
		}
		case FED_CursorType_Selection_e:{
			_FED_LineList_Node_t *LineNode0 = _FED_LineList_GetNodeByReference(
				&wed->LineList,
				Cursor->Selection.LineReference[0]
			);
			_FED_LineList_Node_t *LineNode1 = _FED_LineList_GetNodeByReference(
				&wed->LineList,
				Cursor->Selection.LineReference[1]
			);
			_FED_Character_t *Character0 = &_FED_CharacterList_GetNodeByReference(
				&LineNode0->data.data.CharacterList,
				Cursor->Selection.CharacterReference[0]
			)->data.data;
			_FED_Character_t *Character1 = &_FED_CharacterList_GetNodeByReference(
				&LineNode1->data.data.CharacterList,
				Cursor->Selection.CharacterReference[1]
			)->data.data;
			Character0->CursorReference = -1;
			Character1->CursorReference = -1;
			break;
		}
	}
}

static
void FED_ConvertCursorToSelection(
	FED_t *wed,
	FED_CursorReference_t CursorReference,
	FED_LineReference_t LineReference0,
	FED_CharacterReference_t CharacterReference0,
	FED_LineReference_t LineReference1,
	FED_CharacterReference_t CharacterReference1
){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_UnlinkCursorFromCharacters(wed, CursorReference, Cursor);
	_FED_LineList_Node_t *LineNode0 = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference0);
	_FED_LineList_Node_t *LineNode1 = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference1);
	_FED_Character_t *Character0 = &_FED_CharacterList_GetNodeByReference(
		&LineNode0->data.data.CharacterList,
		CharacterReference0
	)->data.data;
	_FED_Character_t *Character1 = &_FED_CharacterList_GetNodeByReference(
		&LineNode1->data.data.CharacterList,
		CharacterReference1
	)->data.data;
	if(LineReference0 == LineReference1 && CharacterReference0 == CharacterReference1){
		/* source and destination is same */
		Cursor->type = FED_CursorType_FreeStyle_e;
		_FED_MoveCursor_NoCleaning(
			CursorReference,
			&Cursor->FreeStyle.LineReference,
			&Cursor->FreeStyle.CharacterReference,
			LineReference0,
			CharacterReference0,
			Character0
		);
		return;
	}
	else{
		_FED_MoveCursor_NoCleaning(
			CursorReference,
			&Cursor->Selection.LineReference[0],
			&Cursor->Selection.CharacterReference[0],
			LineReference0,
			CharacterReference0,
			Character0
		);
		_FED_MoveCursor_NoCleaning(
			CursorReference,
			&Cursor->Selection.LineReference[1],
			&Cursor->Selection.CharacterReference[1],
			LineReference1,
			CharacterReference1,
			Character1
		);
		Cursor->Selection.PreferredWidth = -1;
		Cursor->type = FED_CursorType_Selection_e;
	}
}

static
FED_CursorReference_t FED_cursor_open(FED_t *wed){
	FED_LineReference_t LineReference;
	_FED_LineList_Node_t *LineNode;
	_FED_Line_t *Line;
	_FED_Character_t *Character;
	if(_FED_LineList_usage(&wed->LineList) == 0){
		/* FED doesnt have any line so lets open a line */
		LineReference = _FED_LineList_NewNodeFirst(&wed->LineList);
		LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		Line = &LineNode->data.data;
		Line->TotalWidth = 0;
		Line->IsEndLine = 1;
		_FED_CharacterList_open(&Line->CharacterList);
		Character = &_FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Line->CharacterList.src
		)->data.data;
		Character->CursorReference = -1;
	}
	else{
		LineReference = _FED_LineList_GetNodeByReference(
			&wed->LineList,
			wed->LineList.src
		)->NextNodeReference;
		LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		Line = &LineNode->data.data;
		Character = &_FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Line->CharacterList.src
		)->data.data;
	}

	FED_CursorReference_t CursorReference = _FED_CursorList_NewNodeLast(&wed->CursorList);
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	Cursor->type = FED_CursorType_FreeStyle_e;
	Cursor->FreeStyle.LineReference = LineReference;
	Cursor->FreeStyle.PreferredWidth = -1;
	Cursor->FreeStyle.CharacterReference = Line->CharacterList.src;
	if(Character->CursorReference != -1){
		assert(0);
	}
	Character->CursorReference = CursorReference;
	return CursorReference;
}
static
void FED_cursor_close(FED_t *wed, FED_CursorReference_t CursorReference){
	_FED_CursorList_Node_t *CursorNode = _FED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	FED_Cursor_t *Cursor = &CursorNode->data.data;
	_FED_UnlinkCursorFromCharacters(wed, CursorReference, Cursor);
	_FED_CursorList_unlink(&wed->CursorList, CursorReference);
}

static
void FED_SetLineWidth(FED_t *wed, uint32_t LineWidth){
	if(LineWidth == wed->LineWidth){
		return;
	}
	bool wib = LineWidth < wed->LineWidth;
	wed->LineWidth = LineWidth;
	if(wib){
		FED_LineReference_t LineReference = _FED_LineList_GetNodeFirst(&wed->LineList);
		while(LineReference != wed->LineList.dst){
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_FED_LineIsIncreased(wed, LineReference, LineNode);

			/* maybe _FED_LineIsIncreased is changed pointers so lets renew LineNode */
			LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);

			LineReference = LineNode->NextNodeReference;
		}
	}
	else{
		FED_LineReference_t LineReference = _FED_LineList_GetNodeFirst(&wed->LineList);
		while(LineReference != wed->LineList.dst){
			_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			FED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			_FED_LineIsDecreased(wed, LineReference, LineNode);

			/* both way is same */
			#if FED_set_debug_InvalidLineAccess == 1
				if(_FED_LineList_IsNodeUnlinked(&wed->LineList, LineNode)){
					LineReference = PrevLineReference;
					LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
				}
			#else
				_FED_LineList_Node_t *PrevLineNode = _FED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
				if(PrevLineNode->NextNodeReference != LineReference){
					LineReference = PrevLineReference;
					LineNode = PrevLineNode;
				}
			#endif

			LineReference = LineNode->NextNodeReference;
		}
	}
}

typedef void (*FED_ExportLine_DataCB_t)(fan::vector_t *TextVector, FED_Data_t Data);

static
void FED_ExportLine(
	FED_t *wed,
	FED_LineReference_t LineReference,
	fan::vector_t *TextVector,
	fan::vector_t *CursorVector,
	bool *IsEndLine,
	FED_ExportLine_DataCB_t DataCB
){
	_FED_LineList_Node_t *LineNode = _FED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_FED_Line_t *Line = &LineNode->data.data;
	*IsEndLine = Line->IsEndLine;
	FED_CharacterReference_t CharacterReference = Line->CharacterList.src;
	_FED_Character_t *Character = &_FED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		CharacterReference
	)->data.data;
	uint32_t x = 0;
	while(1){
		if(Character->CursorReference != -1){
			/* there is cursor, lets export it */
			FED_ExportedCursor_t ExportedCursor;
			ExportedCursor.CursorReference = Character->CursorReference;
			ExportedCursor.x = x;
			fan::VEC_handle(CursorVector);
			((FED_ExportedCursor_t *)&CursorVector->ptr[0])[CursorVector->Current] = ExportedCursor;
			CursorVector->Current++;
		}
		CharacterReference = _FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->NextNodeReference;
		if(CharacterReference == Line->CharacterList.dst){
			break;
		}
		Character = &_FED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->data.data;
		DataCB(TextVector, Character->data);
		x++;
	}
}
