struct _ETC_WED_P(t){
  typedef ETC_WED_set_DataType CharacterData_t;

  enum class CursorType : uint8_t{
    FreeStyle,
    Selection
  };

  #define BLL_set_Language 1
  #define BLL_set_BaseLibrary ETC_WED_set_BaseLibrary
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix _CharacterList
  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _CharacterList_NodeReference_t CharacterReference_t;

  #define BLL_set_Language 1
  #define BLL_set_BaseLibrary ETC_WED_set_BaseLibrary
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix _LineList
  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _LineList_NodeReference_t LineReference_t;

  #define BLL_set_Language 1
  #define BLL_set_BaseLibrary ETC_WED_set_BaseLibrary
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix _CursorList
  #define BLL_set_declare_NodeReference 1
  #define BLL_set_declare_rest 0
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _CursorList_NodeReference_t CursorReference_t;

  #define BLL_set_Language 1
  #define BLL_set_BaseLibrary ETC_WED_set_BaseLibrary
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix _CharacterList
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #define BLL_set_NodeData \
    ETC_WED_set_WidthType width; \
    CharacterData_t data; \
    CursorReference_t CursorReference;
  #define BLL_set_debug_InvalidAction ETC_WED_set_debug_InvalidCharacterAccess
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _CharacterList_NodeData_t _Character_t;

  #define BLL_set_Language 1
  #define BLL_set_BaseLibrary ETC_WED_set_BaseLibrary
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix _LineList
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #define BLL_set_NodeData \
    ETC_WED_set_WidthType TotalWidth; \
    bool IsEndLine; \
    _CharacterList_t CharacterList;
  #define BLL_set_debug_InvalidAction ETC_WED_set_debug_InvalidLineAccess
  #define BLL_set_debug_InvalidAction_srcAccess 0
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _LineList_NodeData_t _Line_t;

  #define BLL_set_Language 1
  #define BLL_set_BaseLibrary ETC_WED_set_BaseLibrary
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_prefix _CursorList
  #define BLL_set_declare_NodeReference 0
  #define BLL_set_declare_rest 1
  #define BLL_set_NodeData \
    CursorType type; \
    union{ \
      struct{ \
        LineReference_t LineReference; \
        CharacterReference_t CharacterReference; \
        ETC_WED_set_WidthType PreferredWidth; \
      }FreeStyle; \
      struct{ \
        LineReference_t LineReference[2]; \
        CharacterReference_t CharacterReference[2]; \
        ETC_WED_set_WidthType PreferredWidth; \
      }Selection; \
    };
  #define BLL_set_debug_InvalidAction ETC_WED_set_debug_InvalidCursorAccess
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _CursorList_NodeData_t Cursor_t;

  struct ExportedCursor_t{
    CursorReference_t CursorReference;
    ETC_WED_set_WidthType x;
    uint32_t y; /* TOOD what type to use? */
  };

  _LineList_t LineList;
  _CursorList_t CursorList;

  uint32_t LineHeight; /* TOOD what type to use? */
  ETC_WED_set_WidthType LineWidth;
  uint32_t LineLimit;  /* TOOD what type to use? */
  uint32_t LineCharacterLimit; /* TOOD what type to use? */

  private:
    #include "private.h"
  public:
    #include "public.h"
};
