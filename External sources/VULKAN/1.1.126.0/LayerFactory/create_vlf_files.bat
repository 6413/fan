REM Copyright (c) 2015-2019 LunarG, Inc. 

REM This file will create a VLF *.def and *.json file in support of a VLF layer
REM
REM ARG1 must be the project name -- $(ProjectName)

set VLF_NAME=%1

set JSON_FILENAME=VkLayer_%VLF_NAME%.json
set DEF_FILENAME=VkLayer_%VLF_NAME%.def

@echo { > %JSON_FILENAME%
@echo     "file_format_version" : "1.1.0", >> %JSON_FILENAME%
@echo     "layer" : { >> %JSON_FILENAME%
@echo         "name": "VK_LAYER_LUNARG_%VLF_NAME%", >> %JSON_FILENAME%
@echo         "type": "GLOBAL", >> %JSON_FILENAME%
@echo         "library_path": ".\\VkLayer_%VLF_NAME%.dll", >> %JSON_FILENAME%
@echo         "api_version": "1.1.126", >> %JSON_FILENAME%
@echo         "implementation_version": "1", >> %JSON_FILENAME%
@echo         "description": "LunarG Validation Factory Layer", >> %JSON_FILENAME%
@echo         "instance_extensions": [ >> %JSON_FILENAME%
@echo              { >> %JSON_FILENAME%
@echo                  "name": "VK_EXT_debug_report", >> %JSON_FILENAME%
@echo                  "spec_version": "6" >> %JSON_FILENAME%
@echo              } >> %JSON_FILENAME%
@echo          ], >> %JSON_FILENAME%
@echo         "device_extensions": [ >> %JSON_FILENAME%
@echo              { >> %JSON_FILENAME%
@echo                  "name": "VK_EXT_debug_marker", >> %JSON_FILENAME%
@echo                  "spec_version": "4", >> %JSON_FILENAME%
@echo                  "entrypoints": ["vkDebugMarkerSetObjectTagEXT", >> %JSON_FILENAME%
@echo                         "vkDebugMarkerSetObjectNameEXT", >> %JSON_FILENAME%
@echo                         "vkCmdDebugMarkerBeginEXT", >> %JSON_FILENAME%
@echo                         "vkCmdDebugMarkerEndEXT", >> %JSON_FILENAME%
@echo                         "vkCmdDebugMarkerInsertEXT" >> %JSON_FILENAME%
@echo                        ] >> %JSON_FILENAME%
@echo              } >> %JSON_FILENAME%
@echo          ] >> %JSON_FILENAME%
@echo     } >> %JSON_FILENAME%
@echo } >> %JSON_FILENAME%

@echo LIBRARY VkLayer_%VLF_NAME% > %DEF_FILENAME%
@echo EXPORTS >> %DEF_FILENAME%
@echo vkGetInstanceProcAddr >> %DEF_FILENAME%
@echo vkGetDeviceProcAddr >> %DEF_FILENAME%
@echo vkEnumerateInstanceLayerProperties >> %DEF_FILENAME%
@echo vkEnumerateInstanceExtensionProperties >> %DEF_FILENAME%
