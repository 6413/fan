void
_RemoveLine(
  LineReference_t LineReference
){
  auto Line = &this->LineList[LineReference];
  Line->CharacterList.Close();
  this->LineList.unlrec(LineReference);
}

void
_MoveCharacterToBeginOfLine(
  _Line_t *srcLine, LineReference_t srcLineReference,
  CharacterReference_t srcCharacterReference, _Character_t *srcCharacter,
  _Line_t *dstLine, LineReference_t dstLineReference
){
  CharacterReference_t dstCharacterReference = dstLine->CharacterList.NewNode();
  dstLine->CharacterList.linkNext(dstLine->CharacterList.src, dstCharacterReference);
  _Character_t *dstCharacter = &dstLine->CharacterList[dstCharacterReference];
  if(srcCharacter->CursorReference.iic() == false){
    Cursor_t *Cursor = &this->CursorList[srcCharacter->CursorReference];
    Cursor->FreeStyle.CharacterReference = dstCharacterReference;
    Cursor->FreeStyle.LineReference = dstLineReference;
  }
  srcLine->TotalWidth -= srcCharacter->width;
  dstLine->TotalWidth += srcCharacter->width;
  dstCharacter->CursorReference = srcCharacter->CursorReference;
  dstCharacter->width = srcCharacter->width;
  dstCharacter->data = srcCharacter->data;
  srcLine->CharacterList.unlrec(srcCharacterReference);
}
void _MoveCharacterToEndOfLine(
  _Line_t *srcLine, LineReference_t srcLineReference,
  CharacterReference_t srcCharacterReference, _Character_t *srcCharacter,
  _Line_t *dstLine, LineReference_t dstLineReference
){
  CharacterReference_t dstCharacterReference = dstLine->CharacterList.NewNode();
  dstLine->CharacterList.linkPrev(dstLine->CharacterList.dst, dstCharacterReference);
  _Character_t *dstCharacter = &dstLine->CharacterList[dstCharacterReference];
  if(srcCharacter->CursorReference.iic() == false){
    Cursor_t *Cursor = &this->CursorList[srcCharacter->CursorReference];
    Cursor->FreeStyle.CharacterReference = dstCharacterReference;
    Cursor->FreeStyle.LineReference = dstLineReference;
  }
  srcLine->TotalWidth -= srcCharacter->width;
  dstLine->TotalWidth += srcCharacter->width;
  dstCharacter->CursorReference = srcCharacter->CursorReference;
  dstCharacter->width = srcCharacter->width;
  dstCharacter->data = srcCharacter->data;
  srcLine->CharacterList.unlrec(srcCharacterReference);
}

void _MoveCursorFreeStyle(
  CursorReference_t CursorReference,
  Cursor_t *Cursor,
  LineReference_t dstLineReference,
  CharacterReference_t dstCharacterReference,
  _Character_t *dstCharacter
){
  _Line_t *srcLine = &this->LineList[Cursor->FreeStyle.LineReference];
  _Character_t *srcCharacter = &srcLine->CharacterList[Cursor->FreeStyle.CharacterReference];
  srcCharacter->CursorReference.sic();
  if(dstCharacter->CursorReference.iic() == false){
    /* there is already cursor what should we do? */
    ETC_WED_set_Abort();
  }
  dstCharacter->CursorReference = CursorReference;
  Cursor->FreeStyle.LineReference = dstLineReference;
  Cursor->FreeStyle.CharacterReference = dstCharacterReference;
}
void _MoveCursor_NoCleaning(
  CursorReference_t CursorReference,
  LineReference_t *srcLineReference, /* will be changed */
  CharacterReference_t *srcCharacterReference, /* will be changed */
  LineReference_t dstLineReference,
  CharacterReference_t dstCharacterReference,
  _Character_t *dstCharacter
){
  if(dstCharacter->CursorReference.iic() == false){
    /* there is already cursor what should we do? */
    ETC_WED_set_Abort();
  }
  dstCharacter->CursorReference = CursorReference;
  *srcLineReference = dstLineReference;
  *srcCharacterReference = dstCharacterReference;
}
void _MoveCursorSelection(
  CursorReference_t CursorReference,
  LineReference_t *srcLineReference, /* will be changed */
  CharacterReference_t *srcCharacterReference, /* will be changed */
  _Character_t *srcCharacter,
  LineReference_t dstLineReference,
  CharacterReference_t dstCharacterReference,
  _Character_t *dstCharacter
){
  srcCharacter->CursorReference.sic();
  _MoveCursor_NoCleaning(
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
void _MoveAllCursors(
  LineReference_t srcLineReference,
  CharacterReference_t srcCharacterReference,
  _Character_t *srcCharacter,
  LineReference_t dstLineReference,
  CharacterReference_t dstCharacterReference,
  _Character_t *dstCharacter
){
  CursorReference_t CursorReference = srcCharacter->CursorReference;
  if(CursorReference.iic() == true){
    /* source doesnt have any cursor */
    return;
  }
  srcCharacter->CursorReference.sic();
  if(dstCharacter->CursorReference.iic() == false){
    /* there is already cursor what should we do? */
    ETC_WED_set_Abort();
  }
  dstCharacter->CursorReference = CursorReference;
  Cursor_t *Cursor = &this->CursorList[CursorReference];
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      Cursor->FreeStyle.LineReference = dstLineReference;
      Cursor->FreeStyle.CharacterReference = dstCharacterReference;
      break;
    }
    case CursorType::Selection:{
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

void _RemoveCharacter_Safe(
  LineReference_t LineReference,
  _Line_t *Line,
  CharacterReference_t CharacterReference,
  _Character_t *Character
){
  CharacterReference_t dstCharacterReference = Line->CharacterList.GetNodeByReference(CharacterReference)->PrevNodeReference;
  _Character_t *dstCharacter = &Line->CharacterList[dstCharacterReference];
  _MoveAllCursors(LineReference, CharacterReference, Character, LineReference, dstCharacterReference, dstCharacter);

  Line->TotalWidth -= Character->width;
  Line->CharacterList.unlrec(CharacterReference);
}
void _RemoveCharacter_Unsafe(
  _Line_t *Line,
  CharacterReference_t CharacterReference,
  _Character_t *Character
){
  Line->TotalWidth -= Character->width;
  Line->CharacterList.unlrec(CharacterReference);
}

/* returns 0 if left is possible */
bool _GetLineAndCharacterOfLeft(
  LineReference_t srcLineReference,
  _LineList_Node_t *srcLineNode,
  CharacterReference_t srcCharacterReference,
  LineReference_t *dstLineReference,
  _Line_t **dstLine,
  CharacterReference_t *dstCharacterReference
){
  _Line_t *srcLine = &srcLineNode->data;
  if(srcCharacterReference == srcLine->CharacterList.src){
    /* its begin of line. can we go up? */
    LineReference_t PrevLineReference = srcLineNode->PrevNodeReference;
    if(PrevLineReference != this->LineList.src){
      _Line_t *PrevLine = &this->LineList[PrevLineReference];
      *dstLineReference = PrevLineReference;
      *dstLine = PrevLine;
      *dstCharacterReference = PrevLine->CharacterList.GetNodeLast();
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
    *dstCharacterReference = srcLine->CharacterList.GetNodeByReference(srcCharacterReference)->PrevNodeReference;
    return 0;
  }
}
void _GetLineAndCharacterOfLeft_Unsafe(
  LineReference_t srcLineReference,
  _LineList_Node_t *srcLineNode,
  CharacterReference_t srcCharacterReference,
  LineReference_t *dstLineReference,
  _Line_t **dstLine,
  CharacterReference_t *dstCharacterReference
){
  _Line_t *srcLine = &srcLineNode->data;
  if(srcCharacterReference == srcLine->CharacterList.src){
    /* its begin of line. can we go up? */
    LineReference_t PrevLineReference = srcLineNode->PrevNodeReference;
    _Line_t *PrevLine = &this->LineList[PrevLineReference];
    *dstLineReference = PrevLineReference;
    *dstLine = PrevLine;
    *dstCharacterReference = PrevLine->CharacterList.GetNodeLast();
  }
  else{
    *dstLineReference = srcLineReference;
    *dstLine = srcLine;
    *dstCharacterReference = srcLine->CharacterList.GetNodeByReference(srcCharacterReference)->PrevNodeReference;
  }
}
/* returns 0 if right is possible */
bool _GetLineAndCharacterOfRight(
  LineReference_t srcLineReference,
  _LineList_Node_t *srcLineNode,
  CharacterReference_t srcCharacterReference,
  LineReference_t *dstLineReference,
  _Line_t **dstLine,
  CharacterReference_t *dstCharacterReference
){
  _Line_t *srcLine = &srcLineNode->data;
  CharacterReference_t NextCharacterReference = srcLine->CharacterList.GetNodeByReference(srcCharacterReference)->NextNodeReference;
  if(NextCharacterReference == srcLine->CharacterList.dst){
    /* its end of line. can we go up? */
    LineReference_t NextLineReference = srcLineNode->NextNodeReference;
    if(NextLineReference != this->LineList.dst){
      _Line_t *NextLine = &this->LineList[NextLineReference];
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
void _GetLineAndCharacterOfRight_Unsafe(
  LineReference_t srcLineReference,
  _LineList_Node_t *srcLineNode,
  CharacterReference_t srcCharacterReference,
  LineReference_t *dstLineReference,
  _Line_t **dstLine,
  CharacterReference_t *dstCharacterReference
){
  _Line_t *srcLine = &srcLineNode->data;
  CharacterReference_t NextCharacterReference = srcLine->CharacterList.GetNodeByReference(srcCharacterReference)->NextNodeReference;
  if(NextCharacterReference == srcLine->CharacterList.dst){
    /* its end of line */
    LineReference_t NextLineReference = srcLineNode->NextNodeReference;
    _Line_t *NextLine = &this->LineList[NextLineReference];
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

/* this function able to change this->LineList pointers */
/* returns 0 if success */
bool _OpenExtraLine(
  LineReference_t LineReference,
  LineReference_t *NextLineReference,
  _LineList_Node_t **NextLineNode
){
  if(this->LineList.Usage() == this->LineLimit){
    if(this->LineList.GetNodeLast() == LineReference){
      return 1;
    }
  }
  *NextLineReference = this->LineList.NewNode();
  this->LineList.linkNext(LineReference, *NextLineReference);
  *NextLineNode = this->LineList.GetNodeByReference(*NextLineReference);
  _Line_t *NextLine = &(*NextLineNode)->data;
  NextLine->TotalWidth = 0;
  _Line_t *Line = &this->LineList[LineReference];
  if(Line->IsEndLine){
    Line->IsEndLine = 0;
    NextLine->IsEndLine = 1;
  }
  else{
    NextLine->IsEndLine = 0;
  }
  NextLine->CharacterList.Open();
  _Character_t *GodCharacter = &NextLine->CharacterList[NextLine->CharacterList.src];
  GodCharacter->CursorReference.sic();
  if(this->LineList.Usage() > this->LineLimit){
    LineReference_t LastLineReference = this->LineList.GetNodeLast();
    _LineList_Node_t *LastLineNode = this->LineList.GetNodeByReference(LastLineReference);
    _Line_t *LastLine = &LastLineNode->data;
    LineReference_t LastPrevLineReference = LastLineNode->PrevNodeReference;
    _Line_t *LastPrevLine = &this->LineList[LastPrevLineReference];

    /* no need to check is it had earlier or not */
    LastPrevLine->IsEndLine = 1;

    CharacterReference_t LastPrevLineLastCharacterReference = LastPrevLine->CharacterList.GetNodeLast();
    _Character_t *LastPrevLineLastCharacter = &LastPrevLine->CharacterList[LastPrevLineLastCharacterReference];
    CharacterReference_t LastLineCharacterReference = LastLine->CharacterList.src;
    while(LastLineCharacterReference != LastLine->CharacterList.dst){
      _CharacterList_Node_t *LastLineCharacterNode = LastLine->CharacterList.GetNodeByReference(LastLineCharacterReference);
      _Character_t *LastLineCharacter = &LastLineCharacterNode->data;
      _MoveAllCursors(LastLineReference, LastLineCharacterReference, LastLineCharacter, LastPrevLineReference, LastPrevLineLastCharacterReference, LastPrevLineLastCharacter);
      LastLineCharacterReference = LastLineCharacterNode->NextNodeReference;
    }
    _RemoveLine(LastLineReference);
  }
  return 0;
}

bool
_IsLineMembersFit(
  uint32_t CharacterAmount, /* TOOD need type */
  ETC_WED_set_WidthType WidthAmount
){
  if(CharacterAmount > this->LineCharacterLimit){
    return 0;
  }
  if(WidthAmount > this->LineWidth){
    return 0;
  }
  return 1;
}
bool _IsLineFit(
  _Line_t *Line
){
  return _IsLineMembersFit(Line->CharacterList.Usage(), Line->TotalWidth);
}
bool _CanCharacterFitLine(
  _Line_t *Line,
  _Character_t *Character
){
  return _IsLineMembersFit(Line->CharacterList.Usage() + 1, Line->TotalWidth + Character->width);
}
CharacterReference_t _LastCharacterReferenceThatFitsToLine(LineReference_t LineReference){
  _Line_t *Line = &this->LineList[LineReference];
  uint32_t CharacterAmount = Line->CharacterList.Usage();
  uint32_t WidthAmount = Line->TotalWidth;
  CharacterReference_t CharacterReference = Line->CharacterList.GetNodeLast();
  while(1){
    if(CharacterReference == Line->CharacterList.src){
      return CharacterReference;
    }
    _CharacterList_Node_t *CharacterNode = Line->CharacterList.GetNodeByReference(CharacterReference);
    _Character_t *Character = &CharacterNode->data;
    if(_IsLineMembersFit(CharacterAmount, WidthAmount)){
      return CharacterReference;
    }
    CharacterAmount -= 1;
    WidthAmount -= Character->width;
    CharacterReference = CharacterNode->PrevNodeReference;
  }
}

void _SlideToNext(LineReference_t LineReference, _LineList_Node_t *LineNode){
  begin:
  LineReference_t NextLineReference;
  _LineList_Node_t *NextLineNode;
  bool IsLoopEntered = !_IsLineFit(&LineNode->data);
  while(!_IsLineFit(&LineNode->data)){
    if(LineNode->data.IsEndLine){
      /* if line has endline we need to create new line to slide */
      if(_OpenExtraLine(LineReference, &NextLineReference, &NextLineNode)){
        CharacterReference_t dstCharacterReference = _LastCharacterReferenceThatFitsToLine(LineReference);
        _Character_t *dstCharacter = &LineNode->data.CharacterList[dstCharacterReference];
        CharacterReference_t srcCharacterReference = LineNode->data.CharacterList.GetNodeLast();
        do{
          _CharacterList_Node_t *srcCharacterNode = LineNode->data.CharacterList.GetNodeByReference(srcCharacterReference);
          _MoveAllCursors(LineReference, srcCharacterReference, &srcCharacterNode->data, LineReference, dstCharacterReference, dstCharacter);
          CharacterReference_t srcCharacterReference_temp = srcCharacterNode->PrevNodeReference;
          _RemoveCharacter_Unsafe(&LineNode->data, srcCharacterReference, &srcCharacterNode->data);
          srcCharacterReference = srcCharacterReference_temp;
        }while(srcCharacterReference != dstCharacterReference);
        return;
      }
      /* that function maybe changed line address so lets get it again */
      LineNode = this->LineList.GetNodeByReference(LineReference);
    }
    else{
      NextLineReference = LineNode->NextNodeReference;
      NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
    }
    CharacterReference_t CharacterReference = LineNode->data.CharacterList.GetNodeLast();
    _Character_t *Character = &LineNode->data.CharacterList[CharacterReference];
    _MoveCharacterToBeginOfLine(
      &LineNode->data,
      LineReference,
      CharacterReference,
      Character,
      &NextLineNode->data,
      NextLineReference
    );
  }
  if(IsLoopEntered){
    LineReference = NextLineReference;
    LineNode = NextLineNode;
    goto begin;
  }
}

void _LineSlideBackFromNext(
  LineReference_t LineReference,
  _LineList_Node_t *LineNode,
  LineReference_t NextLineReference,
  _LineList_Node_t *NextLineNode
){
  Begin:
  CharacterReference_t NextCharacterReference = NextLineNode->data.CharacterList.GetNodeFirst();
  if(NextCharacterReference == NextLineNode->data.CharacterList.dst){
    CharacterReference_t LastCharacterReference = LineNode->data.CharacterList.GetNodeLast();
    _Character_t *LastCharacter = &LineNode->data.CharacterList[LastCharacterReference];
    NextCharacterReference = NextLineNode->data.CharacterList.src;
    _Character_t *NextCharacter = &NextLineNode->data.CharacterList[NextCharacterReference];
    _MoveAllCursors(NextLineReference, NextCharacterReference, NextCharacter, LineReference, LastCharacterReference, LastCharacter);
    bool IsEndLine = NextLineNode->data.IsEndLine;
    _RemoveLine(NextLineReference);
    if(NextLineNode->data.IsEndLine){
      LineNode->data.IsEndLine = 1;
      return;
    }
    else{
      NextLineReference = LineNode->NextNodeReference;
      NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
      goto Begin;
    }
  }
  _CharacterList_Node_t *NextCharacterNode = NextLineNode->data.CharacterList.GetNodeByReference(NextCharacterReference);
  _Character_t *NextCharacter = &NextCharacterNode->data;
  if(_CanCharacterFitLine(&LineNode->data, NextCharacter)){
    _MoveCharacterToEndOfLine(&NextLineNode->data, NextLineReference, NextCharacterReference, NextCharacter, &LineNode->data, LineReference);

    /* TODO this only needed to be processed per line */
    CharacterReference_t LastCharacterReference = LineNode->data.CharacterList.GetNodeLast();
    _Character_t *LastCharacter = &LineNode->data.CharacterList[LastCharacterReference];
    NextCharacterReference = NextLineNode->data.CharacterList.src;
    NextCharacterNode = NextLineNode->data.CharacterList.GetNodeByReference(NextCharacterReference);
    NextCharacter = &NextCharacterNode->data;
    _MoveAllCursors(NextLineReference, NextCharacterReference, NextCharacter, LineReference, LastCharacterReference, LastCharacter);

    goto Begin;
  }
  if(NextLineNode->data.IsEndLine){
    return;
  }
  LineReference = NextLineReference;
  LineNode = NextLineNode;
  NextLineReference = LineNode->NextNodeReference;
  NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
  goto Begin;
}

void _LineIsDecreased(LineReference_t LineReference, _LineList_Node_t *LineNode){
  LineReference_t PrevLineReference = LineNode->PrevNodeReference;
  if(PrevLineReference != this->LineList.src){
    _LineList_Node_t *PrevLineNode = this->LineList.GetNodeByReference(PrevLineReference);
    if(PrevLineNode->data.IsEndLine){
      goto ToNext;
    }
    _LineSlideBackFromNext(PrevLineReference, PrevLineNode, LineReference, LineNode);
  }
  else{
    ToNext:
    if(LineNode->data.IsEndLine){
      return;
    }
    LineReference_t NextLineReference = LineNode->NextNodeReference;
    _LineList_Node_t *NextLineNode = this->LineList.GetNodeByReference(NextLineReference);
    _LineSlideBackFromNext(LineReference, LineNode, NextLineReference, NextLineNode);
  }
}

void _LineIsIncreased(LineReference_t LineReference, _LineList_Node_t *LineNode){
  _LineIsDecreased(LineReference, LineNode);
  _SlideToNext(LineReference, LineNode);
}

void _CursorIsTriggered(Cursor_t *Cursor){
  /* this function must be called when something is changed or could change */
  switch(Cursor->type){
    case CursorType::FreeStyle:{
      Cursor->FreeStyle.PreferredWidth = -1;
      break;
    }
    case CursorType::Selection:{
      Cursor->Selection.PreferredWidth = -1;
      break;
    }
  }
}

void _GetCharacterFromLineByWidth(
  _Line_t *Line,
  uint32_t Width,
  CharacterReference_t *pCharacterReference,
  _Character_t **pCharacter
){
  uint32_t iWidth = 0;
  CharacterReference_t CharacterReference = Line->CharacterList.src;
  while(1){
    CharacterReference_t NextCharacterReference = Line->CharacterList.GetNodeByReference(CharacterReference)->NextNodeReference;
    if(NextCharacterReference == Line->CharacterList.dst){
      /* lets return Character */
      *pCharacter = &Line->CharacterList[CharacterReference];
      *pCharacterReference = CharacterReference;
      return;
    }
    else{
      _Character_t *NextCharacter = &Line->CharacterList[NextCharacterReference];
      if((iWidth + NextCharacter->width) >= Width){
        /* we need to return Character or NextCharacter depends about how close to Width */
        uint32_t CurrentDiff = Width - iWidth;
        uint32_t NextDiff = iWidth + NextCharacter->width - Width;
        if(CurrentDiff <= NextDiff){
          /* lets return Character */
          *pCharacter = &Line->CharacterList[CharacterReference];
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

uint32_t _CalculatePositionOfCharacterInLine(_Line_t *Line, CharacterReference_t pCharacterReference){
  uint32_t Width = 0;
  CharacterReference_t CharacterReference = Line->CharacterList.GetNodeFirst();
  while(1){
    _CharacterList_Node_t *CharacterNode = Line->CharacterList.GetNodeByReference(CharacterReference);
    _Character_t *Character = &CharacterNode->data;
    Width += Character->width;
    if(CharacterReference == pCharacterReference){
      break;
    }
    CharacterReference = CharacterNode->NextNodeReference;
  }
  return Width;
}

void _CalculatePreferredWidthIfNeeded(
  _Line_t *Line,
  CharacterReference_t CharacterReference,
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
    *PreferredWidth = _CalculatePositionOfCharacterInLine(
      Line,
      CharacterReference
    );
  }
}

void _CursorConvertFreeStyleToSelection(
  CursorReference_t CursorReference,
  Cursor_t *Cursor,
  LineReference_t LineReference,
  _Line_t *Line,
  CharacterReference_t CharacterReference,
  _Character_t *Character,
  uint32_t PreferredWidth
){
  Cursor->type = CursorType::Selection;
  Cursor->Selection.PreferredWidth = PreferredWidth;
  Cursor->Selection.LineReference[0] = Cursor->FreeStyle.LineReference;
  Cursor->Selection.CharacterReference[0] = Cursor->FreeStyle.CharacterReference;
  Cursor->Selection.LineReference[1] = LineReference;
  Cursor->Selection.CharacterReference[1] = CharacterReference;
  if(Character->CursorReference.iic() == false){
    /* what will happen to cursor? */
    ETC_WED_set_Abort();
  }
  Character->CursorReference = CursorReference;
}

void _CursorConvertSelectionToFreeStyle(Cursor_t *Cursor, bool Direction){
  Cursor->type = CursorType::FreeStyle;
  {
    LineReference_t LineReference = Cursor->Selection.LineReference[Direction ^ 1];
    _Line_t *Line = &this->LineList[LineReference];
    CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction ^ 1];
    _Character_t *Character = &Line->CharacterList[CharacterReference];
    Character->CursorReference.sic();
  }
  LineReference_t LineReference = Cursor->Selection.LineReference[Direction];
  CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction];
  uint32_t PreferredWidth = Cursor->Selection.PreferredWidth;
  Cursor->FreeStyle.LineReference = LineReference;
  Cursor->FreeStyle.CharacterReference = CharacterReference;
  Cursor->FreeStyle.PreferredWidth = PreferredWidth;
}

void _CursorDeleteSelectedAndMakeCursorFreeStyle(Cursor_t *Cursor){
  bool Direction;
  LineReference_t LineReference0;
  LineReference_t LineReference;
  if(Cursor->Selection.LineReference[0] != Cursor->Selection.LineReference[1]){
    /* lets compare lines */
    if(this->LineList.IsNodeReferenceFronter(
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
    _Line_t *Line = &this->LineList[Cursor->Selection.LineReference[0]];
    if(Line->CharacterList.IsNodeReferenceFronter(
      Cursor->Selection.CharacterReference[0],
      Cursor->Selection.CharacterReference[1])
    ){
      Direction = 1;
    }
    else{
      Direction = 0;
    }
  }
  CharacterReference_t CharacterReference0 = Cursor->Selection.CharacterReference[Direction];
  CharacterReference_t CharacterReference = Cursor->Selection.CharacterReference[Direction ^ 1];
  _Line_t *Line0 = &this->LineList[LineReference0];
  _CursorConvertSelectionToFreeStyle(Cursor, Direction);
  while(1){
    _LineList_Node_t *LineNode = this->LineList.GetNodeByReference(LineReference);
    _Line_t *Line = &LineNode->data;
    _Character_t *Character = &Line->CharacterList[CharacterReference];
    _Character_t *Character0 = &Line0->CharacterList[CharacterReference0];
    _MoveAllCursors(
      LineReference,
      CharacterReference,
      Character,
      LineReference0,
      CharacterReference0,
      Character0
    );
    LineReference_t dstLineReference;
    _Line_t *dstLine;
    CharacterReference_t dstCharacterReference;
    if(CharacterReference != Line->CharacterList.src){
      _GetLineAndCharacterOfLeft_Unsafe(
        LineReference,
        LineNode,
        CharacterReference,
        &dstLineReference,
        &dstLine,
        &dstCharacterReference
      );
      _RemoveCharacter_Unsafe(Line, CharacterReference, Character);
      if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
        /* we did reach where we go */
        _LineIsDecreased(LineReference, LineNode);
        return;
      }
    }
    else{
      /* we need to delete something from previous line */
      BeginOfCharacterReferenceIsFirst:
      dstLineReference = LineNode->PrevNodeReference;
      dstLine = &this->LineList[dstLineReference];
      dstCharacterReference = dstLine->CharacterList.GetNodeLast();
      if(dstLine->IsEndLine == 1){
        if(dstLine->CharacterList.Usage() == 0){
          /* lets delete line */
          _Character_t *srcGodCharacter = &Line->CharacterList[Line->CharacterList.src];
          _Character_t *dstGodCharacter = &dstLine->CharacterList[dstLine->CharacterList.src];
          _MoveAllCursors(
            dstLineReference,
            dstCharacterReference,
            dstGodCharacter,
            LineReference,
            CharacterReference,
            srcGodCharacter
          );
          _RemoveLine(dstLineReference);
          if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
            /* we did reach where we go */
            _LineIsDecreased(LineReference, LineNode);
            return;
          }
          goto BeginOfCharacterReferenceIsFirst;
        }
        else{
          dstLine->IsEndLine = 0;
          if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
            /* we did reach where we go */
            _LineIsDecreased(LineReference, LineNode);
            return;
          }
        }
      }
      else{
        /* nothing to delete */
        if(dstLineReference == LineReference0 && dstCharacterReference == CharacterReference0){
          /* we did reach where we go */
          _LineIsDecreased(LineReference, LineNode);
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
