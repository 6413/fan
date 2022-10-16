void
open(
  uint32_t LineHeight,
  uint32_t LineWidth,
  uint32_t LineLimit,
  uint32_t LineCharacterLimit
){
  this->LineList.Open();
  this->CursorList.Open();
  this->LineHeight = LineHeight;
  this->LineWidth = LineWidth;
  this->LineLimit = LineLimit;
  this->LineCharacterLimit = LineCharacterLimit;
}
void
close(
){
  this->CursorList.Open();
  LineReference_t LineReference = this->LineList.GetNodeFirst();
  while(LineReference != this->LineList.dst){
    _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
    _Line_t *Line = &LineNode->data;
    Line->CharacterList.Open();
    LineReference = LineNode->NextNodeReference;
  }
  this->LineList.Close();
}

/* O(LineNumber) */
LineReference_t
GetLineReferenceByLineIndex(
  uint32_t LineNumber
){
  if(LineNumber >= this->LineList.Usage()){
    LineNumber = this->LineList.Usage() - 1;
  }
  LineReference_t LineReference = this->LineList.GetNodeByReference(this->LineList.src)->NextNodeReference;
  for(; LineNumber; LineNumber--){
    LineReference = this->LineList.GetNodeByReference(LineReference)->NextNodeReference;
  }
  return LineReference;
}

/* O(LineReference) */
uint32_t
GetLineIndexByLineReference(
  LineReference_t LineReference,
  LineReference_t StartLineReference
){
  uint32_t r = 0;
  while(StartLineReference != LineReference){
    StartLineReference = this->LineList.GetNodeByReference(StartLineReference)->NextNodeReference;
    r++;
  }
  return r;
}
uint32_t
GetLineIndexByLineReference(
  LineReference_t LineReference
){
  return GetLineIndexByLineReference(LineReference, this->LineList.GetNodeFirst());
}

/* O(LineReference) */
uint32_t
GetCharacterIndexByCharacterReference(
  LineReference_t LineReference,
  CharacterReference_t CharacterReference,
  CharacterReference_t StartCharacterReference
){
  auto Line = &this->LineList[LineReference];
  uint32_t r = 0;
  while(StartCharacterReference != CharacterReference){
    StartCharacterReference = Line->CharacterList.GetNodeByReference(StartCharacterReference)->NextNodeReference;
    r++;
  }
  return r;
}
uint32_t
GetCharacterIndexByCharacterReference(
  LineReference_t LineReference,
  CharacterReference_t CharacterReference
){
  return GetCharacterIndexByCharacterReference(
    LineReference,
    CharacterReference,
    this->LineList[LineReference].CharacterList.src);
}

void
EndLine(
  CursorReference_t CursorReference
){
  if(this->LineList.Usage() == this->LineLimit){
    return;
  }
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      break;
    }
    case CursorType::Selection:{
      _CursorDeleteSelectedAndMakeCursorFreeStyle(Cursor);
      break;
    }
  }
  LineReference_t LineReference = Cursor->FreeStyle.LineReference;
  LineReference_t NextLineReference = this->LineList.NewNode();
  _Line_t *Line = &this->LineList[LineReference];
  if(Line->IsEndLine == 0){
    Line->IsEndLine = 1;
    this->LineList.linkNext(LineReference, NextLineReference);
    _LineList_Node_t *NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
    _Line_t *NextLine = &NextLineNode->data;
    NextLine->TotalWidth = 0;
    NextLine->IsEndLine = 0;
    NextLine->CharacterList.Open();
    CharacterReference_t CharacterReference = Line->CharacterList.GetNodeByReference(Cursor->FreeStyle.CharacterReference)->NextNodeReference;
    while(CharacterReference != Line->CharacterList.dst){
      _Character_t *Character = &Line->CharacterList[CharacterReference];
      _MoveCharacterToEndOfLine(
        Line,
        LineReference,
        CharacterReference,
        Character,
        NextLine,
        NextLineReference
      );
      CharacterReference = Line->CharacterList.GetNodeByReference(Cursor->FreeStyle.CharacterReference)->NextNodeReference;
    }
    _Character_t *NextLineFirstCharacter = &NextLine->CharacterList[NextLine->CharacterList.src];
    NextLineFirstCharacter->CursorReference.sic();
    _MoveCursorFreeStyle(
      CursorReference,
      Cursor,
      NextLineReference,
      NextLine->CharacterList.src,
      NextLineFirstCharacter
    );
    _LineIsDecreased(NextLineReference, NextLineNode);
  }
  else{
    this->LineList.linkNext(LineReference, NextLineReference);
    _Line_t *NextLine = &this->LineList[NextLineReference];
    NextLine->TotalWidth = 0;
    NextLine->IsEndLine = 1;
    NextLine->CharacterList.Open();
    CharacterReference_t CharacterReference = Line->CharacterList.GetNodeByReference(Cursor->FreeStyle.CharacterReference)->NextNodeReference;
    while(CharacterReference != Line->CharacterList.dst){
      _Character_t *Character = &Line->CharacterList[CharacterReference];
      _MoveCharacterToEndOfLine(
        Line,
        LineReference,
        CharacterReference,
        Character,
        NextLine,
        NextLineReference
      );
      CharacterReference = Line->CharacterList.GetNodeByReference(Cursor->FreeStyle.CharacterReference)->NextNodeReference;
    }
    _Character_t *NextLineFirstCharacter = &NextLine->CharacterList[NextLine->CharacterList.src];
    NextLineFirstCharacter->CursorReference.sic();
    _MoveCursorFreeStyle(
      CursorReference,
      Cursor,
      NextLineReference,
      NextLine->CharacterList.src,
      NextLineFirstCharacter
    );
  }
}

void MoveCursorFreeStyleToLeft(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
      if(_GetLineAndCharacterOfLeft(
        LineReference,
        LineNode,
        CharacterReference,
        &LineReference,
        &Line,
        &CharacterReference)
      ){
        return;
      }
      _Character_t *Character = &Line->CharacterList[CharacterReference];
      _MoveCursorFreeStyle(CursorReference, Cursor, LineReference, CharacterReference, Character);
      break;
    }
    case CursorType::Selection:{
      _CursorConvertSelectionToFreeStyle(Cursor, 1);
      break;
    }
  }
}
void MoveCursorFreeStyleToRight(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line;
      CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
      if(_GetLineAndCharacterOfRight(
        LineReference,
        LineNode,
        CharacterReference,
        &LineReference,
        &Line,
        &CharacterReference)
      ){
        return;
      }
      _Character_t *Character = &Line->CharacterList[CharacterReference];
      _MoveCursorFreeStyle(CursorReference, Cursor, LineReference, CharacterReference, Character);
      break;
    }
    case CursorType::Selection:{
      _CursorConvertSelectionToFreeStyle(Cursor, 0);
      break;
    }
  }
}

void MoveCursorSelectionToLeft(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
      LineReference_t LeftLineReference;
      _Line_t *LeftLine;
      if(_GetLineAndCharacterOfLeft(
        LineReference,
        LineNode,
        CharacterReference,
        &LeftLineReference,
        &LeftLine,
        &CharacterReference
      )){
        return;
      }
      _Character_t *LeftCharacter = &LeftLine->CharacterList[CharacterReference];
      _CursorConvertFreeStyleToSelection(
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
    case CursorType::Selection:{
      LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
      CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(*LineReference);
      _Line_t *Line = &LineNode->data;
      LineReference_t LeftLineReference;
      _Line_t *LeftLine;
      CharacterReference_t LeftCharacterReference;
      if(_GetLineAndCharacterOfLeft(
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
        _CursorConvertSelectionToFreeStyle(Cursor, 0);
        return;
      }
      _Character_t *Character = &Line->CharacterList[*CharacterReference];
      _Character_t *LeftCharacter = &LeftLine->CharacterList[LeftCharacterReference];
      _MoveCursorSelection(
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
void MoveCursorSelectionToRight(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
      LineReference_t RightLineReference;
      _Line_t *RightLine;
      if(_GetLineAndCharacterOfRight(
        LineReference,
        LineNode,
        CharacterReference,
        &RightLineReference,
        &RightLine,
        &CharacterReference
      )){
        return;
      }
      _Character_t *RightCharacter = &RightLine->CharacterList[CharacterReference];
      _CursorConvertFreeStyleToSelection(
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
    case CursorType::Selection:{
      LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
      CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(*LineReference);
      _Line_t *Line = &LineNode->data;
      LineReference_t RightLineReference;
      _Line_t *RightLine;
      CharacterReference_t RightCharacterReference;
      if(_GetLineAndCharacterOfRight(
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
        _CursorConvertSelectionToFreeStyle(Cursor, 0);
        return;
      }
      _Character_t *Character = &Line->CharacterList[*CharacterReference];
      _Character_t *RightCharacter = &RightLine->CharacterList[RightCharacterReference];
      _MoveCursorSelection(
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

void MoveCursorFreeStyleToUp(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      _CalculatePreferredWidthIfNeeded(
        Line,
        Cursor->FreeStyle.CharacterReference,
        &Cursor->FreeStyle.PreferredWidth
      );
      LineReference_t PrevLineReference = LineNode->PrevNodeReference;
      if(PrevLineReference == this->LineList.src){
        /* we already in top */
        return;
      }
      _Line_t *PrevLine = &this->LineList[PrevLineReference];
      CharacterReference_t dstCharacterReference;
      _Character_t *dstCharacter;
      _GetCharacterFromLineByWidth(
        PrevLine,
        Cursor->FreeStyle.PreferredWidth,
        &dstCharacterReference,
        &dstCharacter
      );
      _MoveCursorFreeStyle(CursorReference, Cursor, PrevLineReference, dstCharacterReference, dstCharacter);
      break;
    }
    case CursorType::Selection:{
      _CursorConvertSelectionToFreeStyle(Cursor, 1);
      break;
    }
  }
}
void MoveCursorFreeStyleToDown(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      _CalculatePreferredWidthIfNeeded(
        Line,
        Cursor->FreeStyle.CharacterReference,
        &Cursor->FreeStyle.PreferredWidth
      );
      LineReference_t NextLineReference = LineNode->NextNodeReference;
      if(NextLineReference == this->LineList.dst){
        /* we already in bottom */
        return;
      }
      _Line_t *NextLine = &this->LineList[NextLineReference];
      CharacterReference_t dstCharacterReference;
      _Character_t *dstCharacter;
      _GetCharacterFromLineByWidth(
        NextLine,
        Cursor->FreeStyle.PreferredWidth,
        &dstCharacterReference,
        &dstCharacter
      );
      _MoveCursorFreeStyle(CursorReference, Cursor, NextLineReference, dstCharacterReference, dstCharacter);
      break;
    }
    case CursorType::Selection:{
      _CursorConvertSelectionToFreeStyle(Cursor, 0);
      break;
    }
  }
}

void MoveCursorSelectionToUp(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      _CalculatePreferredWidthIfNeeded(
        Line,
        Cursor->FreeStyle.CharacterReference,
        &Cursor->FreeStyle.PreferredWidth
      );
      LineReference_t PrevLineReference = LineNode->PrevNodeReference;
      if(PrevLineReference == this->LineList.src){
        /* we already in top */
        return;
      }
      _Line_t *PrevLine = &this->LineList[PrevLineReference];
      CharacterReference_t dstCharacterReference;
      _Character_t *dstCharacter;
      _GetCharacterFromLineByWidth(
        PrevLine,
        Cursor->FreeStyle.PreferredWidth,
        &dstCharacterReference,
        &dstCharacter
      );
      _CursorConvertFreeStyleToSelection(
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
    case CursorType::Selection:{
      LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
      CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(*LineReference);
      _Line_t *Line = &LineNode->data;
      _CalculatePreferredWidthIfNeeded(
        Line,
        *CharacterReference,
        &Cursor->Selection.PreferredWidth
      );
      LineReference_t PrevLineReference = LineNode->PrevNodeReference;
      if(PrevLineReference == this->LineList.src){
        /* we already in top */
        return;
      }
      _Line_t *PrevLine = &this->LineList[PrevLineReference];
      CharacterReference_t dstCharacterReference;
      _Character_t *dstCharacter;
      _GetCharacterFromLineByWidth(
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
        _CursorConvertSelectionToFreeStyle(Cursor, 0);
        return;
      }
      _Character_t *Character = &Line->CharacterList[*CharacterReference];
      _MoveCursorSelection(
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
void MoveCursorSelectionToDown(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      _CalculatePreferredWidthIfNeeded(
        Line,
        Cursor->FreeStyle.CharacterReference,
        &Cursor->FreeStyle.PreferredWidth
      );
      LineReference_t NextLineReference = LineNode->NextNodeReference;
      if(NextLineReference == this->LineList.dst){
        /* we already in bottom */
        return;
      }
      _Line_t *NextLine = &this->LineList[NextLineReference];
      CharacterReference_t dstCharacterReference;
      _Character_t *dstCharacter;
      _GetCharacterFromLineByWidth(
        NextLine,
        Cursor->FreeStyle.PreferredWidth,
        &dstCharacterReference,
        &dstCharacter
      );
      _CursorConvertFreeStyleToSelection(
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
    case CursorType::Selection:{
      LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
      CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(*LineReference);
      _Line_t *Line = &LineNode->data;
      _CalculatePreferredWidthIfNeeded(
        Line,
        *CharacterReference,
        &Cursor->Selection.PreferredWidth
      );
      LineReference_t NextLineReference = LineNode->NextNodeReference;
      if(NextLineReference == this->LineList.dst){
        /* we already in bottom */
        return;
      }
      _LineList_Node_t *NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
      _Line_t *NextLine = &this->LineList[NextLineReference];
      CharacterReference_t dstCharacterReference;
      _Character_t *dstCharacter;
      _GetCharacterFromLineByWidth(
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
        _CursorConvertSelectionToFreeStyle(Cursor, 0);
        return;
      }
      _Character_t *Character = &Line->CharacterList[*CharacterReference];
      _MoveCursorSelection(
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

void AddCharacterToCursor(CursorReference_t CursorReference, CharacterData_t data, uint16_t width){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      break;
    }
    case CursorType::Selection:{
      _CursorDeleteSelectedAndMakeCursorFreeStyle(Cursor);
      break;
    }
  }
  LineReference_t LineReference = Cursor->FreeStyle.LineReference;
  _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
  _Line_t *Line = &LineNode->data;
  CharacterReference_t CharacterReference = Line->CharacterList.NewNode();
  Line->CharacterList.linkNext(Cursor->FreeStyle.CharacterReference, CharacterReference);
  _Character_t *Character = &Line->CharacterList[CharacterReference];
  Character->CursorReference.sic();
  Character->width = width;
  Character->data = data;
  Line->TotalWidth += width;
  _MoveCursorFreeStyle(CursorReference, Cursor, LineReference, CharacterReference, Character);
  _LineIsIncreased(LineReference, LineNode);
}

void DeleteCharacterFromCursor(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      CharacterReference_t CharacterReference = Cursor->FreeStyle.CharacterReference;
      if(CharacterReference == Line->CharacterList.src){
        /* nothing to delete but can we delete something from previous line? */
        LineReference_t PrevLineReference = LineNode->PrevNodeReference;
        if(PrevLineReference != this->LineList.src){
          _LineList_Node_t *PrevLineNode = this->LineList.GetNodeByReference(PrevLineReference);
          _Line_t *PrevLine = &PrevLineNode->data;
          if(PrevLine->IsEndLine){
            _Character_t *GodCharacter = &Line->CharacterList[Line->CharacterList.src];
            CharacterReference_t PrevLineLastCharacterReference = PrevLine->CharacterList.GetNodeLast();
            _Character_t *PrevLineLastCharacter = &PrevLine->CharacterList[PrevLineLastCharacterReference];
            _MoveAllCursors(
              LineReference,
              Line->CharacterList.src,
              GodCharacter,
              PrevLineReference,
              PrevLineLastCharacterReference,
              PrevLineLastCharacter
            );
            PrevLine->IsEndLine = 0;
            _LineIsDecreased(PrevLineReference, PrevLineNode);
            return;
          }
          else{
            CharacterReference = PrevLine->CharacterList.GetNodeLast();
            _Character_t *Character = &PrevLine->CharacterList[CharacterReference];
            _RemoveCharacter_Safe(
              PrevLineReference,
              PrevLine,
              CharacterReference,
              Character
            );
            _LineIsDecreased(PrevLineReference, PrevLineNode);
          }
        }
        else{
          /* previous line doesnt exists at all */
          return;
        }
      }
      else{
        _Character_t *Character = &Line->CharacterList[CharacterReference];
        _RemoveCharacter_Safe(LineReference, Line, CharacterReference, Character);
        _LineIsDecreased(LineReference, LineNode);
      }
      break;
    }
    case CursorType::Selection:{
      _CursorDeleteSelectedAndMakeCursorFreeStyle(Cursor);
      break;
    }
  }
}

void DeleteCharacterFromCursorRight(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _Line_t *Line = &LineNode->data;
      CharacterReference_t CharacterReference = Line->CharacterList.GetNodeByReference(Cursor->FreeStyle.CharacterReference)->NextNodeReference;
      if(CharacterReference == Line->CharacterList.dst){
        /* we are in end of line */
        LineReference_t NextLineReference = LineNode->NextNodeReference;
        if(Line->IsEndLine == 1){
          /* lets delete endline */
          if(NextLineReference != this->LineList.dst){
            _Line_t *NextLine = &this->LineList[NextLineReference];
            if(NextLine->CharacterList.Usage()){
              Line->IsEndLine = 0;
              _LineIsDecreased(LineReference, LineNode);
            }
            else{
              /* next line doesnt have anything so lets just Unlink it */
              _Character_t *NextLineGodCharacter = &NextLine->CharacterList[NextLine->CharacterList.src];
              _Character_t *Character = &Line->CharacterList[CharacterReference];
              _MoveAllCursors(
                NextLineReference,
                NextLine->CharacterList.src,
                NextLineGodCharacter,
                LineReference,
                CharacterReference,
                Character
              );
              _RemoveLine(NextLineReference);
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
          _LineList_Node_t *NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
          _Line_t *NextLine = &NextLineNode->data;
          CharacterReference = NextLine->CharacterList.GetNodeFirst();
          _Character_t *Character = &Line->CharacterList[CharacterReference];
          _RemoveCharacter_Safe(LineReference, Line, CharacterReference, Character);
          _LineIsDecreased(NextLineReference, NextLineNode);
        }
      }
      else{
        _Character_t *Character = &Line->CharacterList[CharacterReference];
        _RemoveCharacter_Safe(LineReference, Line, CharacterReference, Character);
        _LineIsDecreased(LineReference, LineNode);
      }
      break;
    }
    case CursorType::Selection:{
      _CursorDeleteSelectedAndMakeCursorFreeStyle(Cursor);
      break;
    }
  }
}

void MoveCursorFreeStyleToBeginOfLine(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      break;
    }
    case CursorType::Selection:{
      _CursorConvertSelectionToFreeStyle(Cursor, 1);
      break;
    }
  }
  _Line_t *Line = &this->LineList[Cursor->FreeStyle.LineReference];
  _Character_t *Character = &Line->CharacterList[Cursor->FreeStyle.CharacterReference];
  Character->CursorReference.sic();
  CharacterReference_t BeginCharacterReference = Line->CharacterList.src;
  _Character_t *BeginCharacter = &Line->CharacterList[BeginCharacterReference];
  if(BeginCharacter->CursorReference.iic() == false){
    /* there is already cursor there */
    ETC_WED_set_Abort();
  }
  BeginCharacter->CursorReference = CursorReference;
  Cursor->FreeStyle.CharacterReference = BeginCharacterReference;
}
void MoveCursorFreeStyleToEndOfLine(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      break;
    }
    case CursorType::Selection:{
      _CursorConvertSelectionToFreeStyle(Cursor, 1);
      break;
    }
  }
  _Line_t *Line = &this->LineList[Cursor->FreeStyle.LineReference];
  _Character_t *Character = &Line->CharacterList[Cursor->FreeStyle.CharacterReference];
  Character->CursorReference.sic();
  CharacterReference_t EndCharacterReference = Line->CharacterList.GetNodeLast();
  _Character_t *EndCharacter = &Line->CharacterList[EndCharacterReference];
  if(EndCharacter->CursorReference.iic() == false){
    /* there is already cursor there */
    ETC_WED_set_Abort();
  }
  EndCharacter->CursorReference = CursorReference;
  Cursor->FreeStyle.CharacterReference = EndCharacterReference;
}

void MoveCursorSelectionToBeginOfLine(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _Line_t *Line = &this->LineList[LineReference];
      CharacterReference_t FirstCharacterReference = Line->CharacterList.src;
      _Character_t *FirstCharacter = &Line->CharacterList[FirstCharacterReference];
      _CursorConvertFreeStyleToSelection(
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
    case CursorType::Selection:{
      LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
      CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
      _Line_t *Line = &this->LineList[*LineReference];
      _Character_t *Character = &Line->CharacterList[*CharacterReference];
      CharacterReference_t FirstCharacterReference = Line->CharacterList.src;
      if(
        Cursor->Selection.CharacterReference[0] == FirstCharacterReference &&
        Cursor->Selection.LineReference[0] == *LineReference
      ){
        /* where we went is same with where we started */
        /* so lets convert cursor to FreeStyle back */
        _CursorConvertSelectionToFreeStyle(Cursor, 0);
        return;
      }
      _Character_t *FirstCharacter = &Line->CharacterList[FirstCharacterReference];
      _MoveCursorSelection(
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
void MoveCursorSelectionToEndOfLine(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _CursorIsTriggered(Cursor);
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      LineReference_t LineReference = Cursor->FreeStyle.LineReference;
      _Line_t *Line = &this->LineList[LineReference];
      CharacterReference_t LastCharacterReference = Line->CharacterList.GetNodeLast();
      _Character_t *LastCharacter = &Line->CharacterList[LastCharacterReference];
      _CursorConvertFreeStyleToSelection(
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
    case CursorType::Selection:{
      LineReference_t *LineReference = &Cursor->Selection.LineReference[1];
      CharacterReference_t *CharacterReference = &Cursor->Selection.CharacterReference[1];
      _Line_t *Line = &this->LineList[*LineReference];
      _Character_t *Character = &Line->CharacterList[*CharacterReference];
      CharacterReference_t LastCharacterReference = Line->CharacterList.GetNodeLast();
      if(
        Cursor->Selection.CharacterReference[0] == LastCharacterReference &&
        Cursor->Selection.LineReference[0] == *LineReference
      ){
        /* where we went is same with where we started */
        /* so lets convert cursor to FreeStyle back */
        _CursorConvertSelectionToFreeStyle(Cursor, 0);
        return;
      }
      _Character_t *LastCharacter = &Line->CharacterList[LastCharacterReference];
      _MoveCursorSelection(
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

void GetLineAndCharacter(
  LineReference_t HintLineReference,
  uint32_t y,
  uint32_t x,
  LineReference_t *LineReference, /* w */
  CharacterReference_t *CharacterReference /* w */
){
  y /= this->LineHeight;
  while(y--){
    HintLineReference = this->LineList.GetNodeByReference(HintLineReference)->NextNodeReference;
    if(HintLineReference == this->LineList.dst){
      HintLineReference = this->LineList.GetNodeByReference(HintLineReference)->PrevNodeReference;
      x = 0xffffffff;
      break;
    }
  }
  _Line_t *Line = &this->LineList[HintLineReference];
  _Character_t *unused;
  _GetCharacterFromLineByWidth(Line, x, CharacterReference, &unused);
  *LineReference = HintLineReference;
}

void _UnlinkCursorFromCharacters(CursorReference_t CursorReference, Cursor_t *Cursor){
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      _Line_t *Line = &this->LineList[Cursor->FreeStyle.LineReference];
      _Character_t *Character = &Line->CharacterList[Cursor->FreeStyle.CharacterReference];
      Character->CursorReference.sic();
      break;
    }
    case CursorType::Selection:{
      _Line_t *Line0 = &this->LineList[Cursor->Selection.LineReference[0]];
      _Line_t *Line1 = &this->LineList[Cursor->Selection.LineReference[1]];
      _Character_t *Character0 = &Line0->CharacterList[Cursor->Selection.CharacterReference[0]];
      _Character_t *Character1 = &Line1->CharacterList[Cursor->Selection.CharacterReference[1]];
      Character0->CursorReference.sic();
      Character1->CursorReference.sic();
      break;
    }
  }
}

void ConvertCursorToSelection(
  CursorReference_t CursorReference,
  LineReference_t LineReference0,
  CharacterReference_t CharacterReference0,
  LineReference_t LineReference1,
  CharacterReference_t CharacterReference1
){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _UnlinkCursorFromCharacters(CursorReference, Cursor);
  _Line_t *Line0 = &this->LineList[LineReference0];
  _Line_t *Line1 = &this->LineList[LineReference1];
  _Character_t *Character0 = &Line0->CharacterList[CharacterReference0];
  _Character_t *Character1 = &Line1->CharacterList[CharacterReference1];
  if(LineReference0 == LineReference1 && CharacterReference0 == CharacterReference1){
    /* source and destination is same */
    Cursor->type = CursorType::FreeStyle;
    _MoveCursor_NoCleaning(
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
    _MoveCursor_NoCleaning(
      CursorReference,
      &Cursor->Selection.LineReference[0],
      &Cursor->Selection.CharacterReference[0],
      LineReference0,
      CharacterReference0,
      Character0
    );
    _MoveCursor_NoCleaning(
      CursorReference,
      &Cursor->Selection.LineReference[1],
      &Cursor->Selection.CharacterReference[1],
      LineReference1,
      CharacterReference1,
      Character1
    );
    Cursor->Selection.PreferredWidth = -1;
    Cursor->type = CursorType::Selection;
  }
}

CursorReference_t
cursor_open(
){
  LineReference_t LineReference;
  _Line_t *Line;
  _Character_t *Character;
  if(this->LineList.Usage() == 0){
    /* WED doesnt have any line so lets open a line */
    LineReference = this->LineList.NewNodeFirst();
    Line = &this->LineList[LineReference];
    Line->TotalWidth = 0;
    Line->IsEndLine = 1;
    Line->CharacterList.Open();
    Character = &Line->CharacterList[Line->CharacterList.src];
    Character->CursorReference.sic();
  }
  else{
    LineReference = this->LineList.GetNodeByReference(this->LineList.src)->NextNodeReference;
    Line = &this->LineList[LineReference];
    Character = &Line->CharacterList[Line->CharacterList.src];
  }

  CursorReference_t CursorReference = this->CursorList.NewNodeLast();
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  Cursor->type = CursorType::FreeStyle;
  Cursor->FreeStyle.LineReference = LineReference;
  Cursor->FreeStyle.PreferredWidth = -1;
  Cursor->FreeStyle.CharacterReference = Line->CharacterList.src;
  if(Character->CursorReference.iic() == false){
    ETC_WED_set_Abort();
  }
  Character->CursorReference = CursorReference;
  return CursorReference;
}
void cursor_close(CursorReference_t CursorReference){
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  _UnlinkCursorFromCharacters(CursorReference, Cursor);
  this->CursorList.unlrec(CursorReference);
}

void SetLineWidth(uint32_t LineWidth){
  if(LineWidth == this->LineWidth){
    return;
  }
  bool wib = LineWidth < this->LineWidth;
  this->LineWidth = LineWidth;
  if(wib){
    LineReference_t LineReference = this->LineList.GetNodeFirst();
    while(LineReference != this->LineList.dst){
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      _LineIsIncreased(LineReference, LineNode);

      /* maybe _LineIsIncreased is changed pointers so lets renew LineNode */
      LineNode = this->LineList.GetNodeByReference(LineReference);

      LineReference = LineNode->NextNodeReference;
    }
  }
  else{
    LineReference_t LineReference = this->LineList.GetNodeFirst();
    while(LineReference != this->LineList.dst){
      _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
      LineReference_t PrevLineReference = LineNode->PrevNodeReference;
      _LineIsDecreased(LineReference, LineNode);

      /* both way is same */
      #if set_debug_InvalidLineAccess == 1
        if(this->LineList.IsNodeUnlinked(LineNode)){
          LineReference = PrevLineReference;
          LineNode = this->LineList.GetNodeByReference(LineReference);
        }
      #else
        _LineList_Node_t *PrevLineNode = this->LineList.GetNodeByReference(PrevLineReference);
        if(PrevLineNode->NextNodeReference != LineReference){
          LineReference = PrevLineReference;
          LineNode = PrevLineNode;
        }
      #endif

      LineReference = LineNode->NextNodeReference;
    }
  }
}

struct CursorInformation_t{
  CursorType type;
  union{
    struct{
      LineReference_t LineReference;
      CharacterReference_t CharacterReference;
    }FreeStyle;
    struct{
      LineReference_t LineReference[2];
      CharacterReference_t CharacterReference[2];
    }Selection;
  };
};
void
GetCursorInformation(
  CursorReference_t CursorReference,
  CursorInformation_t *CursorInformation /* w */
){
  auto Cursor = &this->CursorList[CursorReference];
  CursorInformation->type = Cursor->type;
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      CursorInformation->FreeStyle.LineReference = Cursor->FreeStyle.LineReference;
      CursorInformation->FreeStyle.CharacterReference = Cursor->FreeStyle.CharacterReference;
      break;
    }
    case CursorType::Selection:{
      CursorInformation->Selection.LineReference[0] = Cursor->Selection.LineReference[0];
      CursorInformation->Selection.LineReference[1] = Cursor->Selection.LineReference[1];
      CursorInformation->Selection.CharacterReference[0] = Cursor->Selection.CharacterReference[0];
      CursorInformation->Selection.CharacterReference[1] = Cursor->Selection.CharacterReference[1];
      break;
    }
  }
}

struct ExportLine_t{
  _CharacterList_t *CharacterList;
  CharacterReference_t CharacterReference;
};
void
ExportLine_init(
  ExportLine_t *ExportLine,
  LineReference_t LineReference
){
  _Line_t *Line = &this->LineList[LineReference];
  ExportLine->CharacterList = &Line->CharacterList;
  ExportLine->CharacterReference = Line->CharacterList.GetNodeFirst();
}
bool
ExportLine(
  ExportLine_t *ExportLine,
  CharacterReference_t *CharacterReference
){
  if(ExportLine->CharacterReference == ExportLine->CharacterList->dst){
    return false;
  }
  *CharacterReference = ExportLine->CharacterReference;
  ExportLine->CharacterReference = ExportLine->CharacterList->GetNodeByReference(ExportLine->CharacterReference)->NextNodeReference;
  return true;
}

CharacterData_t *
GetDataOfCharacter(
  LineReference_t LineReference,
  CharacterReference_t CharacterReference
){
  _Line_t *Line = &this->LineList[LineReference];
  return &Line->CharacterList[CharacterReference].data;
}

LineReference_t
GetFirstLineID
(
){
  return this->LineList.GetNodeFirst();
}
LineReference_t
GetLastLineID
(
){
  return this->LineList.GetNodeLast();
}
