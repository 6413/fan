#ifndef FED_set_DataType
	#define FED_set_DataType wchar_t
#endif

typedef FED_set_DataType FED_Data_t;

#define BLL_set_BaseLibrary 1
#define BLL_set_prefix _FED_CharacterList
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)
typedef _FED_CharacterList_NodeReference_t FED_CharacterReference_t;

#define BLL_set_BaseLibrary 1
#define BLL_set_prefix _FED_LineList
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)
typedef _FED_LineList_NodeReference_t FED_LineReference_t;

#define BLL_set_BaseLibrary 1
#define BLL_set_prefix _FED_CursorList
#define BLL_set_declare_rest 0
#include _FAN_PATH(BLL/BLL.h)
typedef _FED_CursorList_NodeReference_t FED_CursorReference_t;

typedef struct{
	uint16_t width;
	FED_Data_t data;

	FED_CursorReference_t CursorReference;
}_FED_Character_t;
#define BLL_set_BaseLibrary 1
#define BLL_set_prefix _FED_CharacterList
#define BLL_set_declare_rest 1
#define BLL_set_node_data _FED_Character_t data;
#define BLL_set_debug_InvalidAction FED_set_debug_InvalidCharacterAccess
#include _FAN_PATH(BLL/BLL.h)

typedef struct{
	uint32_t TotalWidth;
	bool IsEndLine;
	_FED_CharacterList_t CharacterList;
}_FED_Line_t;
#define BLL_set_BaseLibrary 1
#define BLL_set_prefix _FED_LineList
#define BLL_set_declare_rest 1
#define BLL_set_node_data _FED_Line_t data;
#define BLL_set_debug_InvalidAction FED_set_debug_InvalidLineAccess
#define BLL_set_debug_InvalidAction_srcAccess 0
#include _FAN_PATH(BLL/BLL.h)

typedef struct{
	uint8_t type;
	union{
		struct{
			FED_LineReference_t LineReference;
			FED_CharacterReference_t CharacterReference;
			uint32_t PreferredWidth;
		}FreeStyle;
		struct{
			FED_LineReference_t LineReference[2];
			FED_CharacterReference_t CharacterReference[2];
			uint32_t PreferredWidth;
		}Selection;
	};
}FED_Cursor_t;
#define BLL_set_BaseLibrary 1
#define BLL_set_prefix _FED_CursorList
#define BLL_set_declare_rest 1
#define BLL_set_node_data FED_Cursor_t data;
#define BLL_set_debug_InvalidAction FED_set_debug_InvalidCursorAccess
#include _FAN_PATH(BLL/BLL.h)

enum{
	FED_CursorType_FreeStyle_e,
	FED_CursorType_Selection_e
};

typedef struct{
	_FED_LineList_t LineList;
	_FED_CursorList_t CursorList;

	uint32_t LineHeight;
	uint32_t LineWidth;
	uint32_t LineLimit;
	uint32_t LineCharacterLimit;
}FED_t;

typedef struct{
	FED_CursorReference_t CursorReference;
	uint32_t x;
	uint32_t y;
}FED_ExportedCursor_t;
