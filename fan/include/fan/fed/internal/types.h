#ifndef WED_set_DataType
	#define WED_set_DataType wchar_t
#endif

typedef WED_set_DataType WED_Data_t;

#define BLL_set_prefix _WED_CharacterList
#define BLL_set_declare_basic_types 1
#define BLL_set_declare_rest 0
#include <WITCH/BLL/BLL.h>
typedef _WED_CharacterList_NodeReference_t WED_CharacterReference_t;

#define BLL_set_prefix _WED_LineList
#define BLL_set_declare_basic_types 1
#define BLL_set_declare_rest 0
#include <WITCH/BLL/BLL.h>
typedef _WED_LineList_NodeReference_t WED_LineReference_t;

#define BLL_set_prefix _WED_CursorList
#define BLL_set_declare_basic_types 1
#define BLL_set_declare_rest 0
#include <WITCH/BLL/BLL.h>
typedef _WED_CursorList_NodeReference_t WED_CursorReference_t;

typedef struct{
	uint16_t width;
	WED_Data_t data;

	WED_CursorReference_t CursorReference;
}_WED_Character_t;
#define BLL_set_prefix _WED_CharacterList
#define BLL_set_declare_basic_types 0
#define BLL_set_declare_rest 1
#define BLL_set_node_data _WED_Character_t data;
#define BLL_set_debug_InvalidAccess WED_set_debug_InvalidCharacterAccess
#include <WITCH/BLL/BLL.h>

typedef struct{
	uint32_t TotalWidth;
	bool IsEndLine;
	_WED_CharacterList_t CharacterList;
}_WED_Line_t;
#define BLL_set_prefix _WED_LineList
#define BLL_set_declare_basic_types 0
#define BLL_set_declare_rest 1
#define BLL_set_node_data _WED_Line_t data;
#define BLL_set_debug_InvalidAccess WED_set_debug_InvalidLineAccess
#include <WITCH/BLL/BLL.h>

typedef struct{
	uint8_t type;
	union{
		struct{
			WED_LineReference_t LineReference;
			WED_CharacterReference_t CharacterReference;
			uint32_t PreferredWidth;
		}FreeStyle;
		struct{
			WED_LineReference_t LineReference[2];
			WED_CharacterReference_t CharacterReference[2];
			uint32_t PreferredWidth;
		}Selection;
	};
}WED_Cursor_t;
#define BLL_set_prefix _WED_CursorList
#define BLL_set_declare_basic_types 0
#define BLL_set_declare_rest 1
#define BLL_set_node_data WED_Cursor_t data;
#define BLL_set_debug_InvalidAccess WED_set_debug_InvalidCursorAccess
#include <WITCH/BLL/BLL.h>

enum{
	WED_CursorType_FreeStyle_e,
	WED_CursorType_Selection_e
};

typedef struct{
	_WED_LineList_t LineList;
	_WED_CursorList_t CursorList;

	uint32_t LineHeight;
	uint32_t LineWidth;
	uint16_t SpaceSize;
}WED_t;

typedef struct{
	WED_CursorReference_t CursorReference;
	uint32_t x;
	uint32_t y;
}WED_ExportedCursor_t;
