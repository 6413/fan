void WED_open(WED_t *wed, uint32_t LineHeight, uint32_t LineWidth){
	_WED_LineList_open(&wed->LineList);
	_WED_CursorList_open(&wed->CursorList);
	wed->LineHeight = LineHeight;
	wed->LineWidth = LineWidth;
}
void WED_close(WED_t *wed){

}

WED_LineReference_t WED_GetLineReferenceByLineIndex(WED_t *wed, uint32_t LineNumber){
	if(LineNumber >= _WED_LineList_usage(&wed->LineList)){
		return -1;
	}
	WED_LineReference_t LineReference = _WED_LineList_GetNodeByReference(
		&wed->LineList,
		wed->LineList.src
	)->NextNodeReference;
	for(; LineNumber; LineNumber--){
		LineReference = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference)->NextNodeReference;
	}
	return LineReference;
}

void WED_EndLine(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
	WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
	WED_LineReference_t NextLineReference = _WED_LineList_NewNode(&wed->LineList);
	_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_WED_Line_t *Line = &LineNode->data.data;
	if(Line->IsEndLine == 0){
		Line->IsEndLine = 1;
		_WED_LineList_linkNext(&wed->LineList, LineReference, NextLineReference);
		_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_WED_Line_t *NextLine = &NextLineNode->data.data;
		NextLine->TotalWidth = 0;
		NextLine->IsEndLine = 0;
		_WED_CharacterList_open(&NextLine->CharacterList);
		_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_WED_Line_t *Line = &LineNode->data.data;
		WED_CharacterReference_t CharacterReference = _WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Cursor->FreeStyle.CharacterReference
		)->NextNodeReference;
		while(CharacterReference != Line->CharacterList.dst){
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_WED_MoveCharacterToEndOfLine(
				wed,
				Line,
				LineReference,
				CharacterReference,
				Character,
				NextLine,
				NextLineReference
			);
			CharacterReference = _WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->NextNodeReference;
		}
		_WED_Character_t *NextLineFirstCharacter = &_WED_CharacterList_GetNodeByReference(
			&NextLine->CharacterList,
			NextLine->CharacterList.src
		)->data.data;
		NextLineFirstCharacter->CursorReference = -1;
		_WED_MoveCursorFreeStyle(
			wed,
			CursorReference,
			Cursor,
			NextLineReference,
			NextLine->CharacterList.src,
			NextLineFirstCharacter
		);
		_WED_LineIsDecreased(wed, NextLineReference, NextLineNode);
	}
	else{
		_WED_LineList_linkNext(&wed->LineList, LineReference, NextLineReference);
		_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
		_WED_Line_t *NextLine = &NextLineNode->data.data;
		NextLine->TotalWidth = 0;
		NextLine->IsEndLine = 1;
		_WED_CharacterList_open(&NextLine->CharacterList);
		_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		_WED_Line_t *Line = &LineNode->data.data;
		WED_CharacterReference_t CharacterReference = _WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Cursor->FreeStyle.CharacterReference
		)->NextNodeReference;
		while(CharacterReference != Line->CharacterList.dst){
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_WED_MoveCharacterToEndOfLine(
				wed,
				Line,
				LineReference,
				CharacterReference,
				Character,
				NextLine,
				NextLineReference
			);
			CharacterReference = _WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->NextNodeReference;
		}
		_WED_Character_t *NextLineFirstCharacter = &_WED_CharacterList_GetNodeByReference(
			&NextLine->CharacterList,
			NextLine->CharacterList.src
		)->data.data;
		NextLineFirstCharacter->CursorReference = -1;
		_WED_MoveCursorFreeStyle(
			wed,
			CursorReference,
			Cursor,
			NextLineReference,
			NextLine->CharacterList.src,
			NextLineFirstCharacter
		);
	}
}

void WED_MoveCursorFreeStyleToLeft(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			if(_WED_GetLineAndCharacterOfLeft(
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
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_WED_MoveCursorFreeStyle(wed, CursorReference, Cursor, LineReference, CharacterReference, Character);
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
}
void WED_MoveCursorFreeStyleToRight(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line;
			WED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			if(_WED_GetLineAndCharacterOfRight(
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
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				CharacterReference
			)->data.data;
			_WED_MoveCursorFreeStyle(wed, CursorReference, Cursor, LineReference, CharacterReference, Character);
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
			break;
		}
	}
}

void WED_MoveCursorSelectionToLeft(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			WED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			WED_LineReference_t LeftLineReference;
			_WED_Line_t *LeftLine;
			if(_WED_GetLineAndCharacterOfLeft(
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
			_WED_Character_t *LeftCharacter = &_WED_CharacterList_GetNodeByReference(
				&LeftLine->CharacterList,
				CharacterReference
			)->data.data;
			_WED_CursorConvertFreeStyleToSelection(
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
		case WED_CursorType_Selection_e:{
			WED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			WED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_LineReference_t LeftLineReference;
			_WED_Line_t *LeftLine;
			WED_CharacterReference_t LeftCharacterReference;
			if(_WED_GetLineAndCharacterOfLeft(
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
				_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_WED_Character_t *LeftCharacter = &_WED_CharacterList_GetNodeByReference(
				&LeftLine->CharacterList,
				LeftCharacterReference
			)->data.data;
			_WED_MoveCursorSelection(
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
void WED_MoveCursorSelectionToRight(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			WED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			WED_LineReference_t RightLineReference;
			_WED_Line_t *RightLine;
			if(_WED_GetLineAndCharacterOfRight(
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
			_WED_Character_t *RightCharacter = &_WED_CharacterList_GetNodeByReference(
				&RightLine->CharacterList,
				CharacterReference
			)->data.data;
			_WED_CursorConvertFreeStyleToSelection(
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
		case WED_CursorType_Selection_e:{
			WED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			WED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_LineReference_t RightLineReference;
			_WED_Line_t *RightLine;
			WED_CharacterReference_t RightCharacterReference;
			if(_WED_GetLineAndCharacterOfRight(
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
				_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_WED_Character_t *RightCharacter = &_WED_CharacterList_GetNodeByReference(
				&RightLine->CharacterList,
				RightCharacterReference
			)->data.data;
			_WED_MoveCursorSelection(
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

void WED_MoveCursorFreeStyleToUp(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference == wed->LineList.src){
				/* we already in top */
				return;
			}
			_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_WED_Line_t *PrevLine = &PrevLineNode->data.data;
			WED_CharacterReference_t dstCharacterReference;
			_WED_Character_t *dstCharacter;
			_WED_GetCharacterFromLineByWidth(
				wed,
				PrevLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_WED_MoveCursorFreeStyle(wed, CursorReference, Cursor, PrevLineReference, dstCharacterReference, dstCharacter);
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
}
void WED_MoveCursorFreeStyleToDown(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			WED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			if(NextLineReference == wed->LineList.dst){
				/* we already in bottom */
				return;
			}
			_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_WED_Line_t *NextLine = &NextLineNode->data.data;
			WED_CharacterReference_t dstCharacterReference;
			_WED_Character_t *dstCharacter;
			_WED_GetCharacterFromLineByWidth(
				wed,
				NextLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_WED_MoveCursorFreeStyle(wed, CursorReference, Cursor, NextLineReference, dstCharacterReference, dstCharacter);
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
			break;
		}
	}
}

void WED_MoveCursorSelectionToUp(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference == wed->LineList.src){
				/* we already in top */
				return;
			}
			_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_WED_Line_t *PrevLine = &PrevLineNode->data.data;
			WED_CharacterReference_t dstCharacterReference;
			_WED_Character_t *dstCharacter;
			_WED_GetCharacterFromLineByWidth(
				wed,
				PrevLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_WED_CursorConvertFreeStyleToSelection(
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
		case WED_CursorType_Selection_e:{
			WED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			WED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				*CharacterReference,
				&Cursor->Selection.PreferredWidth
			);
			WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
			if(PrevLineReference == wed->LineList.src){
				/* we already in top */
				return;
			}
			_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, PrevLineReference);
			_WED_Line_t *PrevLine = &PrevLineNode->data.data;
			WED_CharacterReference_t dstCharacterReference;
			_WED_Character_t *dstCharacter;
			_WED_GetCharacterFromLineByWidth(
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
				_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_WED_MoveCursorSelection(
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
void WED_MoveCursorSelectionToDown(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				Cursor->FreeStyle.CharacterReference,
				&Cursor->FreeStyle.PreferredWidth
			);
			WED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			if(NextLineReference == wed->LineList.dst){
				/* we already in bottom */
				return;
			}
			_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_WED_Line_t *NextLine = &NextLineNode->data.data;
			WED_CharacterReference_t dstCharacterReference;
			_WED_Character_t *dstCharacter;
			_WED_GetCharacterFromLineByWidth(
				wed,
				NextLine,
				Cursor->FreeStyle.PreferredWidth,
				&dstCharacterReference,
				&dstCharacter
			);
			_WED_CursorConvertFreeStyleToSelection(
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
		case WED_CursorType_Selection_e:{
			WED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			WED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_CalculatePreferredWidthIfNeeded(
				wed,
				Line,
				*CharacterReference,
				&Cursor->Selection.PreferredWidth
			);
			WED_LineReference_t NextLineReference = LineNode->NextNodeReference;
			if(NextLineReference == wed->LineList.dst){
				/* we already in bottom */
				return;
			}
			_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(&wed->LineList, NextLineReference);
			_WED_Line_t *NextLine = &NextLineNode->data.data;
			WED_CharacterReference_t dstCharacterReference;
			_WED_Character_t *dstCharacter;
			_WED_GetCharacterFromLineByWidth(
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
				_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			_WED_MoveCursorSelection(
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

void WED_AddCharacterToCursor(WED_t *wed, WED_CursorReference_t CursorReference, WED_Data_t data, uint16_t width){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
	WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
	_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_WED_Line_t *Line = &LineNode->data.data;
	WED_CharacterReference_t CharacterReference = _WED_CharacterList_NewNode(&Line->CharacterList);
	_WED_CharacterList_linkNext(&Line->CharacterList, Cursor->FreeStyle.CharacterReference, CharacterReference);
	_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		CharacterReference
	)->data.data;
	Character->CursorReference = -1;
	Character->width = width;
	Character->data = data;
	Line->TotalWidth += width;
	_WED_MoveCursorFreeStyle(wed, CursorReference, Cursor, LineReference, CharacterReference, Character);
	_WED_LineIsIncreased(wed, LineReference, LineNode);
}

void WED_DeleteCharacterFromCursor(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
			if(CharacterReference == Line->CharacterList.src){
				/* nothing to delete but can we delete something from previous line? */
				WED_LineReference_t PrevLineReference = LineNode->PrevNodeReference;
				if(PrevLineReference != wed->LineList.src){
					_WED_LineList_Node_t *PrevLineNode = _WED_LineList_GetNodeByReference(
						&wed->LineList,
						PrevLineReference
					);
					_WED_Line_t *PrevLine = &PrevLineNode->data.data;
					if(PrevLine->IsEndLine){
						/* lets just delete end line */
						if(_WED_CharacterList_usage(&Line->CharacterList)){
							PrevLine->IsEndLine = 0;
							_WED_LineIsDecreased(wed, PrevLineReference, PrevLineNode);
						}
						else{
							/* previous line doesnt have anything we must unlink it directly */
							_WED_Character_t *GodCharacter = &_WED_CharacterList_GetNodeByReference(
								&Line->CharacterList,
								Line->CharacterList.src
							)->data.data;
							WED_CharacterReference_t PrevLineLastCharacterReference = _WED_CharacterList_GetNodeLast(
								&PrevLine->CharacterList
							);
							_WED_Character_t *PrevLineLastCharacter = &_WED_CharacterList_GetNodeByReference(
								&PrevLine->CharacterList,
								PrevLineLastCharacterReference
							)->data.data;
							_WED_MoveAllCursors(
								wed,
								LineReference,
								Line->CharacterList.src,
								GodCharacter,
								PrevLineReference,
								PrevLineLastCharacterReference,
								PrevLineLastCharacter
							);
							_WED_LineList_unlink(&wed->LineList, LineReference);
						}
						return;
					}
					else{
						CharacterReference = _WED_CharacterList_GetNodeLast(&PrevLine->CharacterList);
						_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
							&PrevLine->CharacterList,
							CharacterReference
						)->data.data;
						_WED_RemoveCharacter_Safe(
							wed,
							PrevLineReference,
							PrevLine,
							CharacterReference,
							Character
						);
						_WED_LineIsDecreased(wed, PrevLineReference, PrevLineNode);
					}
				}
				else{
					/* previous line doesnt exists at all */
					return;
				}
			}
			else{
				_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
					&Line->CharacterList,
					CharacterReference
				)->data.data;
				_WED_RemoveCharacter_Safe(wed, LineReference, Line, CharacterReference, Character);
				_WED_LineIsDecreased(wed, LineReference, LineNode);
			}
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
}

void WED_DeleteCharacterFromCursorRight(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_CharacterReference_t CharacterReference = _WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->NextNodeReference;
			if(CharacterReference == Line->CharacterList.dst){
				/* we are in end of line */
				WED_LineReference_t NextLineReference = LineNode->NextNodeReference;
				if(Line->IsEndLine == 1){
					/* lets delete endline */
					if(NextLineReference != wed->LineList.dst){
						_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(
							&wed->LineList,
							NextLineReference
						);
						_WED_Line_t *NextLine = &NextLineNode->data.data;
						if(_WED_CharacterList_usage(&NextLine->CharacterList)){
							Line->IsEndLine = 0;
							_WED_LineIsDecreased(wed, LineReference, LineNode);
						}
						else{
							/* next line doesnt have anything so lets just unlink it */
							_WED_Character_t *NextLineGodCharacter = &_WED_CharacterList_GetNodeByReference(
								&NextLine->CharacterList,
								NextLine->CharacterList.src
							)->data.data;
							_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
								&Line->CharacterList,
								CharacterReference
							)->data.data;
							_WED_MoveAllCursors(
								wed,
								NextLineReference,
								NextLine->CharacterList.src,
								NextLineGodCharacter,
								LineReference,
								CharacterReference,
								Character
							);
							_WED_LineList_unlink(&wed->LineList, NextLineReference);
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
					_WED_LineList_Node_t *NextLineNode = _WED_LineList_GetNodeByReference(
						&wed->LineList,
						NextLineReference
					);
					_WED_Line_t *NextLine = &NextLineNode->data.data;
					CharacterReference = _WED_CharacterList_GetNodeFirst(&NextLine->CharacterList);
					_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
						&Line->CharacterList,
						CharacterReference
					)->data.data;
					_WED_RemoveCharacter_Safe(wed, LineReference, Line, CharacterReference, Character);
					_WED_LineIsDecreased(wed, NextLineReference, NextLineNode);
				}
			}
			else{
				_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
					&Line->CharacterList,
					CharacterReference
				)->data.data;
				_WED_RemoveCharacter_Safe(wed, LineReference, Line, CharacterReference, Character);
				_WED_LineIsDecreased(wed, LineReference, LineNode);
			}
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorDeleteSelectedAndMakeCursorFreeStyle(wed, Cursor);
			break;
		}
	}
}

void WED_MoveCursorFreeStyleToBeginOfLine(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
	_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, Cursor->FreeStyle.LineReference);
	_WED_Line_t *Line = &LineNode->data.data;
	_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		Cursor->FreeStyle.CharacterReference
	)->data.data;
	Character->CursorReference = -1;
	WED_CharacterReference_t BeginCharacterReference = Line->CharacterList.src;
	_WED_Character_t *BeginCharacter = &_WED_CharacterList_GetNodeByReference(
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
void WED_MoveCursorFreeStyleToEndOfLine(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 1);
			break;
		}
	}
	_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, Cursor->FreeStyle.LineReference);
	_WED_Line_t *Line = &LineNode->data.data;
	_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		Cursor->FreeStyle.CharacterReference
	)->data.data;
	Character->CursorReference = -1;
	WED_CharacterReference_t EndCharacterReference = _WED_CharacterList_GetNodeLast(&Line->CharacterList);
	_WED_Character_t *EndCharacter = &_WED_CharacterList_GetNodeByReference(
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

void WED_MoveCursorSelectionToBeginOfLine(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_CharacterReference_t FirstCharacterReference = Line->CharacterList.src;
			_WED_Character_t *FirstCharacter = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				FirstCharacterReference
			)->data.data;
			_WED_CursorConvertFreeStyleToSelection(
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
		case WED_CursorType_Selection_e:{
			WED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			WED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			WED_CharacterReference_t FirstCharacterReference = Line->CharacterList.src;
			if(
				Cursor->Selection.CharacterReference[0] == FirstCharacterReference &&
				Cursor->Selection.LineReference[0] == *LineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_WED_Character_t *FirstCharacter = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				FirstCharacterReference
			)->data.data;
			_WED_MoveCursorSelection(
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
void WED_MoveCursorSelectionToEndOfLine(WED_t *wed, WED_CursorReference_t CursorReference){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_CursorIsTriggered(Cursor);
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			WED_LineReference_t LineReference = Cursor->FreeStyle.LineReference;
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			WED_CharacterReference_t LastCharacterReference = _WED_CharacterList_GetNodeLast(&Line->CharacterList);
			_WED_Character_t *LastCharacter = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				LastCharacterReference
			)->data.data;
			_WED_CursorConvertFreeStyleToSelection(
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
		case WED_CursorType_Selection_e:{
			WED_LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
			WED_CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, *LineReference);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				*CharacterReference
			)->data.data;
			WED_CharacterReference_t LastCharacterReference = _WED_CharacterList_GetNodeLast(&Line->CharacterList);
			if(
				Cursor->Selection.CharacterReference[0] == LastCharacterReference &&
				Cursor->Selection.LineReference[0] == *LineReference
			){
				/* where we went is same with where we started */
				/* so lets convert cursor to FreeStyle back */
				_WED_CursorConvertSelectionToFreeStyle(wed, Cursor, 0);
				return;
			}
			_WED_Character_t *LastCharacter = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				LastCharacterReference
			)->data.data;
			_WED_MoveCursorSelection(
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

/* returns 1 if not possible */
void WED_GetLineAndCharacter(
	WED_t *wed,
	WED_LineReference_t HintLineReference,
	uint32_t y,
	uint32_t x,
	WED_LineReference_t *LineReference, /* w */
	WED_CharacterReference_t *CharacterReference /* w */
){
	y /= wed->LineHeight;
	while(y--){
		HintLineReference = _WED_LineList_GetNodeByReference(&wed->LineList, HintLineReference)->NextNodeReference;
		if(HintLineReference == wed->LineList.dst){
			HintLineReference = _WED_LineList_GetNodeByReference(&wed->LineList, HintLineReference)->PrevNodeReference;
			x = 0xffffffff;
			break;
		}
	}
	_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, HintLineReference);
	_WED_Line_t *Line = &LineNode->data.data;
	_WED_Character_t *unused;
	_WED_GetCharacterFromLineByWidth(wed, Line, x, CharacterReference, &unused);
	*LineReference = HintLineReference;
}

void _WED_UnlinkCursorFromCharacters(WED_t *wed, WED_CursorReference_t CursorReference, WED_Cursor_t *Cursor){
	switch(Cursor->type){
		case WED_CursorType_FreeStyle_e:{
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(
				&wed->LineList,
				Cursor->FreeStyle.LineReference
			);
			_WED_Line_t *Line = &LineNode->data.data;
			_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
				&Line->CharacterList,
				Cursor->FreeStyle.CharacterReference
			)->data.data;
			Character->CursorReference = -1;
			break;
		}
		case WED_CursorType_Selection_e:{
			_WED_LineList_Node_t *LineNode0 = _WED_LineList_GetNodeByReference(
				&wed->LineList,
				Cursor->Selection.LineReference[0]
			);
			_WED_LineList_Node_t *LineNode1 = _WED_LineList_GetNodeByReference(
				&wed->LineList,
				Cursor->Selection.LineReference[1]
			);
			_WED_Character_t *Character0 = &_WED_CharacterList_GetNodeByReference(
				&LineNode0->data.data.CharacterList,
				Cursor->Selection.CharacterReference[0]
			)->data.data;
			_WED_Character_t *Character1 = &_WED_CharacterList_GetNodeByReference(
				&LineNode1->data.data.CharacterList,
				Cursor->Selection.CharacterReference[1]
			)->data.data;
			Character0->CursorReference = -1;
			Character1->CursorReference = -1;
			break;
		}
	}
}

void WED_ConvertCursorToSelection(
	WED_t *wed,
	WED_CursorReference_t CursorReference,
	WED_LineReference_t LineReference0,
	WED_CharacterReference_t CharacterReference0,
	WED_LineReference_t LineReference1,
	WED_CharacterReference_t CharacterReference1
){
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	_WED_UnlinkCursorFromCharacters(wed, CursorReference, Cursor);
	_WED_LineList_Node_t *LineNode0 = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference0);
	_WED_LineList_Node_t *LineNode1 = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference1);
	_WED_Character_t *Character0 = &_WED_CharacterList_GetNodeByReference(
		&LineNode0->data.data.CharacterList,
		CharacterReference0
	)->data.data;
	_WED_Character_t *Character1 = &_WED_CharacterList_GetNodeByReference(
		&LineNode1->data.data.CharacterList,
		CharacterReference1
	)->data.data;
	if(LineReference0 == LineReference1 && CharacterReference0 == CharacterReference1){
		/* source and destination is same */
		Cursor->type = WED_CursorType_FreeStyle_e;
		_WED_MoveCursor_NoCleaning(
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
		_WED_MoveCursor_NoCleaning(
			CursorReference,
			&Cursor->Selection.LineReference[0],
			&Cursor->Selection.CharacterReference[0],
			LineReference0,
			CharacterReference0,
			Character0
		);
		_WED_MoveCursor_NoCleaning(
			CursorReference,
			&Cursor->Selection.LineReference[1],
			&Cursor->Selection.CharacterReference[1],
			LineReference1,
			CharacterReference1,
			Character1
		);
		Cursor->Selection.PreferredWidth = -1;
		Cursor->type = WED_CursorType_Selection_e;
	}
}

WED_CursorReference_t WED_cursor_open(WED_t *wed){
	WED_LineReference_t LineReference;
	_WED_LineList_Node_t *LineNode;
	_WED_Line_t *Line;
	_WED_Character_t *Character;
	if(_WED_LineList_usage(&wed->LineList) == 0){
		/* WED doesnt have any line so lets open a line */
		LineReference = _WED_LineList_NewNodeFirst(&wed->LineList);
		LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		Line = &LineNode->data.data;
		Line->TotalWidth = 0;
		Line->IsEndLine = 1;
		_WED_CharacterList_open(&Line->CharacterList);
		Character = &_WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			Line->CharacterList.src
		)->data.data;
		Character->CursorReference = -1;
	}
	else{
		LineReference = _WED_LineList_GetNodeByReference(
			&wed->LineList,
			wed->LineList.src
		)->NextNodeReference;
		LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
		Line = &LineNode->data.data;
	}

	WED_CursorReference_t CursorReference = _WED_CursorList_NewNodeLast(&wed->CursorList);
	_WED_CursorList_Node_t *CursorNode = _WED_CursorList_GetNodeByReference(&wed->CursorList, CursorReference);
	WED_Cursor_t *Cursor = &CursorNode->data.data;
	Cursor->type = WED_CursorType_FreeStyle_e;
	Cursor->FreeStyle.LineReference = LineReference;
	Cursor->FreeStyle.PreferredWidth = -1;
	Cursor->FreeStyle.CharacterReference = Line->CharacterList.src;
	if(Character->CursorReference != -1){
		assert(0);
	}
	Character->CursorReference = CursorReference;
	return CursorReference;
}

void WED_SetLineWidth(WED_t *wed, uint32_t LineWidth){
	bool wib = LineWidth < wed->LineWidth;
	wed->LineWidth = LineWidth;
	if(wib){
		WED_LineReference_t LineReference = _WED_LineList_GetNodeFirst(&wed->LineList);
		while(LineReference != wed->LineList.dst){
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_LineIsIncreased(wed, LineReference, LineNode);

			/* maybe _WED_LineIsIncreased is changed pointers so lets renew LineNode */
			LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);

			LineReference = LineNode->NextNodeReference;
		}
	}
	else{
		WED_LineReference_t LineReference = _WED_LineList_GetNodeFirst(&wed->LineList);
		while(LineReference != wed->LineList.dst){
			_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
			_WED_LineIsDecreased(wed, LineReference, LineNode);
			LineReference = LineNode->NextNodeReference;
		}
	}
}

typedef void (*WED_ExportLine_DataCB_t)(VEC_t *TextVector, WED_Data_t Data);

void WED_ExportLine(
	WED_t *wed,
	WED_LineReference_t LineReference,
	VEC_t *TextVector,
	VEC_t *CursorVector,
	bool *IsEndLine,
	WED_ExportLine_DataCB_t DataCB
){
	_WED_LineList_Node_t *LineNode = _WED_LineList_GetNodeByReference(&wed->LineList, LineReference);
	_WED_Line_t *Line = &LineNode->data.data;
	*IsEndLine = Line->IsEndLine;
	WED_CharacterReference_t CharacterReference = Line->CharacterList.src;
	_WED_Character_t *Character = &_WED_CharacterList_GetNodeByReference(
		&Line->CharacterList,
		CharacterReference
	)->data.data;
	uint32_t x = 0;
	while(1){
		if(Character->CursorReference != -1){
			/* there is cursor, lets export it */
			WED_ExportedCursor_t ExportedCursor;
			ExportedCursor.CursorReference = Character->CursorReference;
			ExportedCursor.x = x;
			VEC_pushback(CursorVector, WED_ExportedCursor_t, ExportedCursor);
		}
		CharacterReference = _WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->NextNodeReference;
		if(CharacterReference == Line->CharacterList.dst){
			break;
		}
		Character = &_WED_CharacterList_GetNodeByReference(
			&Line->CharacterList,
			CharacterReference
		)->data.data;
		DataCB(TextVector, Character->data);
		x++;
	}
}
