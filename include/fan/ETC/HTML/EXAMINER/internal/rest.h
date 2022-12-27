#include _WITCH_PATH(MEM/MEM.h)
#include _WITCH_PATH(STR/common/common.h)

#if ETC_HTML_EXAMINER_set_AbortIfBadData == 1
  #include _WITCH_PATH(PR/PR.h)
#endif

#define _P(p0) CONCAT3(ETC_HTML_EXAMINER_set_prefix, _, p0)
#define _PP(p0) CONCAT4(_, ETC_HTML_EXAMINER_set_prefix, _, p0)

enum{
  _P(ExamineError_Done_e),
  _P(ExamineError_BadData_e)
};

enum{
  _P(ResultType_TagDOCTYPE_e),
  _P(ResultType_TagHTML_e),
  _P(ResultType_TagHEAD_e),
  _P(ResultType_TagTITLE_e),
  _P(ResultType_TagLINK_e),
  _P(ResultType_TagMETA_e),
  _P(ResultType_TagSCRIPT_e),
  _P(ResultType_TagBODY_e),
  _P(ResultType_TagDIV_e),
  _P(ResultType_TagBLOCKQUOTE_e),
  _P(ResultType_TagP_e),
  _P(ResultType_Content_e)
};

enum{
  _P(ResultFlagTag_Closing_e) = 0x01
};

typedef struct{
  struct{
    uint8_t *v;
    uintptr_t s;
  }Version;
}_P(Result_TagDOCTYPE_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagHTML_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagHEAD_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagTITLE_t);

typedef struct{
  uint32_t filler;
}_P(Result_TagLINK_t);

typedef struct{
  uint32_t filler;
}_P(Result_TagMETA_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagSCRIPT_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagBODY_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagDIV_t);

typedef struct{
  uint8_t Flag;
}_P(Result_TagBLOCKQUOTE_t);

typedef struct{
  uint32_t filler;
}_P(Result_TagP_t);

typedef struct{
  struct{
    uint8_t *v;
    uintptr_t s;
  }Itself;
}_P(Result_Content_t);

typedef struct{
  union{
    _P(Result_TagDOCTYPE_t) TagDOCTYPE;
    _P(Result_TagHTML_t) TagHTML;
    _P(Result_TagHEAD_t) TagHEAD;
    _P(Result_TagTITLE_t) TagTITLE;
    _P(Result_TagLINK_t) TagLINK;
    _P(Result_TagMETA_t) TagMETA;
    _P(Result_TagSCRIPT_t) TagSCRIPT;
    _P(Result_TagBODY_t) TagBODY;
    _P(Result_TagDIV_t) TagDIV;
    _P(Result_TagBLOCKQUOTE_t) TagBLOCKQUOTE;
    _P(Result_TagP_t) TagP;
    _P(Result_Content_t) Content;
  };
}_P(Result_t);

typedef struct{
  uint32_t filler;
}_P(t);

void
_P(init)
(
  _P(t) *Examiner
){
  
}

bool _PP(ncasecmp)(const void *s0, const void *s1, uintptr_t size){
  return STR_ncasecmp(s0, s1, size) == 0;
}

uint8_t *_PP(findnonspace)(const uint8_t *s0){
  while(*s0 == ' '){
    ++s0;
  }
  return s0;
}

/*
  for negative return values ~ExamineError_*_e
  other returns values are ResultType_*_e
*/
sint32_t
_P(Examine)
(
  _P(t) *Examiner,
  void *Data,
  uintptr_t DataSize,
  uintptr_t *DataIndex,
  _P(Result_t) *Result
){
  uintptr_t LeftSize = DataSize - *DataIndex;
  uintptr_t findchru = MEM_findchru(&((uint8_t *)Data)[*DataIndex], LeftSize, '<');
  if(findchru == LeftSize){
    /* TODO implement later */
    PR_abort();
  }
  else if(findchru != 0){
    Result->Content.Itself.v = &((uint8_t *)Data)[*DataIndex];
    Result->Content.Itself.s = findchru;
    *DataIndex += findchru;
    return _P(ResultType_Content_e);
  }
  --LeftSize;
  uint8_t *Tag = &((uint8_t *)Data)[*DataIndex + 1];
  uintptr_t TagSize = MEM_findchru(Tag, LeftSize, '>');
  uint8_t *TagEnd = &Tag[TagSize];
  if(TagSize == LeftSize){
    #if ETC_HTML_EXAMINER_set_AbortIfBadData == 1
      PR_abort();
    #endif
    return ~_P(ExamineError_BadData_e);
  }
  else if(TagSize == 0){
    /* <> */
    #if ETC_HTML_EXAMINER_set_AbortIfBadData == 1
      PR_abort();
    #endif
    return ~_P(ExamineError_BadData_e);
  }

  /* <TagSize> */
  /* +       + */
  /* 1       1 */
  *DataIndex += 1 + TagSize + 1;

  uint8_t *TagNameEnd = Tag;
  {
    uint8_t *TagEnd = &Tag[TagSize];
    while(TagNameEnd < TagEnd){
      if(*TagNameEnd == ' '){
        break;
      }
      ++TagNameEnd;
    }
  }
  uintptr_t TagNameSize = (uintptr_t)TagNameEnd - (uintptr_t)Tag;
  uint8_t *TagName = Tag;
  uint8_t ResultFlagTag_Closing;
  switch(*TagName){
    case '!':{

    }
    case '/':{
      ResultFlagTag_Closing = _P(ResultFlagTag_Closing_e);
      ++TagName;
      --TagNameSize;
      break;
    }
    default:{
      ResultFlagTag_Closing = 0;
      break;
    }
  }
  switch(TagNameSize){
    case 0x00:{
      /* 
        first character is space like:
        < >
        </>
        </ >
      */
      #if ETC_HTML_EXAMINER_set_AbortIfBadData == 1
        PR_abort();
      #endif
      return ~_P(ExamineError_BadData_e);
    }
    case 0x01:{
      if(_PP(ncasecmp)("p", TagName, 0x01)){
        /* enjoy */
        PR_abort();
      }
      break;
    }
    case 0x03:{
      if(_PP(ncasecmp)("div", TagName, 0x03)){
        Result->TagDIV.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagDIV_e);
      }
      break;
    }
    case 0x04:{
      if(_PP(ncasecmp)("html", TagName, 0x04)){
        Result->TagHTML.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagHTML_e);
      }
      else if(_PP(ncasecmp)("head", TagName, 0x04)){
        Result->TagHEAD.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagHEAD_e);
      }
      else if(_PP(ncasecmp)("link", TagName, 0x04)){
        return _P(ResultType_TagLINK_e);
      }
      else if(_PP(ncasecmp)("meta", TagName, 0x04)){
        return _P(ResultType_TagMETA_e);
      }
      else if(_PP(ncasecmp)("body", TagName, 0x04)){
        Result->TagBODY.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagBODY_e);
      }
      break;
    }
    case 0x05:{
      if(_PP(ncasecmp)("title", TagName, 0x05)){
        Result->TagTITLE.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagTITLE_e);
      }
      break;
    }
    case 0x06:{
      if(_PP(ncasecmp)("script", TagName, 0x06)){
        Result->TagSCRIPT.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagSCRIPT_e);
      }
      break;
    }
    case 0x08:{
      if(_PP(ncasecmp)("!DOCTYPE", TagName, 0x08)){
        uint8_t *VersionPointer = _PP(findnonspace)(TagNameEnd);
        if(VersionPointer == TagEnd){
          Result->TagDOCTYPE.Version.v = 0;
          Result->TagDOCTYPE.Version.s = 0;
        }
        else{
          Result->TagDOCTYPE.Version.v = VersionPointer;
          Result->TagDOCTYPE.Version.s = (uintptr_t)TagEnd - (uintptr_t)VersionPointer;
        }
        return _P(ResultType_TagDOCTYPE_e);
      }
      break;
    }
    case 0x0a:{
      if(_PP(ncasecmp)("blockquote", TagName, 0x0a)){
        Result->TagBLOCKQUOTE.Flag = ResultFlagTag_Closing;
        return _P(ResultType_TagBLOCKQUOTE_e);
      }
      break;
    }
  }
  /* unknown Tag name */
  #if ETC_HTML_EXAMINER_set_AbortIfBadData == 1
    WriteInformation("%u \"%.*s\"\n", TagNameSize, TagNameSize, Tag);
    PR_abort();
  #endif
  return ~_P(ExamineError_BadData_e);
}

#undef _P
#undef _PP
