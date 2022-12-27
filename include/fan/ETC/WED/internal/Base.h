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
    uint16_t width; \
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
    uint32_t TotalWidth; \
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
        uint32_t PreferredWidth; \
      }FreeStyle; \
      struct{ \
        LineReference_t LineReference[2]; \
        CharacterReference_t CharacterReference[2]; \
        uint32_t PreferredWidth; \
      }Selection; \
    };
  #define BLL_set_debug_InvalidAction ETC_WED_set_debug_InvalidCursorAccess
  #include _ETC_WED_INCLUDE(BLL/BLL.h)
  typedef _CursorList_NodeData_t Cursor_t;

  struct ExportedCursor_t{
    CursorReference_t CursorReference;
    uint32_t x;
    uint32_t y;
  };

  _LineList_t LineList;
  _CursorList_t CursorList;

  uint32_t LineHeight;
  uint32_t LineWidth;
  uint32_t LineLimit;
  uint32_t LineCharacterLimit;

  private:
    #include "private.h"
  public:
    #include "public.h"
};
