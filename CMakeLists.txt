cmake_minimum_required(VERSION 3.28)

set(FAN_PHYSICS OFF)
set(FAN_GUI ON)
set(FAN_JSON ON)
set(FAN_3D OFF)
set(FAN_OPENGL ON)
set(FAN_VULKAN OFF)
set(FAN_FMT ON)

set(BUILD_MAIN ON)

#set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1)
#set(CMAKE_NINJA_OUTPUT_PATH_PREFIX "")

set(CMAKE_COLOR_DIAGNOSTICS ON)

#set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
#set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if (FAN_GUI)
	# set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
# endif()
# set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
# if (FAN_GUI)
	# set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
# endif()
# #set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
# set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 2)


find_program(CCACHE_PROGRAM ccache NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
if(CCACHE_PROGRAM)
		set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=gold")

set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "/usr/bin/clang-scan-deps")
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_MAP_FLAG "-fmodule-mapper=")
set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "-fmodule-only")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fprebuilt-module-path=${CMAKE_BINARY_DIR}/CMakeFiles/fan_modules.dir")

project(fan)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/fan/)
set(OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/fan)
set(THIRDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/fan)

file(MAKE_DIRECTORY ${OUTPUT_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${THIRDPARTY_PATH}/include)
link_directories(${THIRDPARTY_PATH}/lib)

if(UNIX AND FAN_GUI)
	set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH OFF)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)
endif()

set(BUILD_SHARED_LIBS ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")


function(apply_fan_definitions target)
	target_compile_definitions(${target} PRIVATE
		$<$<BOOL:${FAN_OPENGL}>:fan_opengl>
		$<$<BOOL:${FAN_GUI}>:fan_gui>
		$<$<BOOL:${FAN_JSON}>:fan_json>
		$<$<BOOL:${FAN_3D}>:fan_3D>
		$<$<BOOL:${FAN_PHYSICS}>:fan_physics>
		$<$<BOOL:${FAN_VULKAN}>:fan_vulkan>
		loco_line
		loco_rectangle
		loco_sprite
		loco_light
		loco_circle
		loco_responsive_text
		loco_universal_image_renderer
		IMGUI_IMPL_OPENGL_LOADER_CUSTOM
	)
endfunction()

function(apply_common_compile_options target)
	target_compile_options(${target} PRIVATE
		-MD -MT -Wall -Wextra -ferror-limit=2 -stdlib=libstdc++ -fsized-deallocation
		-Wno-shift-op-parentheses -Wno-unused-variable -Wno-int-to-void-pointer-cast
		-Wno-unused-parameter -Wno-unused-function -Wno-bitwise-op-parentheses
		-Wno-invalid-offsetof -Wno-missing-field-initializers -Wno-sign-compare
		-Wno-unused-but-set-parameter -Wno-unused-value -pthread -w
	)
endfunction()

#add_library(fan_pch INTERFACE)
#target_precompile_headers(fan_pch INTERFACE ${ROOT_PATH}/pch.h)


add_library(fan_modules STATIC)
apply_fan_definitions(fan_modules)



set(MODULE_FILES
	${ROOT_PATH}/print.ixx
	${ROOT_PATH}/types/color.ixx
	${ROOT_PATH}/types/vector.ixx
	${ROOT_PATH}/types/quaternion.ixx
	${ROOT_PATH}/types/matrix.ixx
	${ROOT_PATH}/types/magic.ixx
	${ROOT_PATH}/types/masterpiece.ixx
	${ROOT_PATH}/types/traits.ixx
	
	${ROOT_PATH}/types/fstring.ixx
	
 	${ROOT_PATH}/random.ixx
	
	${ROOT_PATH}/io/directory.ixx
	${ROOT_PATH}/io/file.ixx
	
	${ROOT_PATH}/graphics/common_types.ixx
	
	${ROOT_PATH}/graphics/camera.ixx
	${ROOT_PATH}/graphics/image_load.ixx
	${ROOT_PATH}/graphics/webp.ixx
	${ROOT_PATH}/graphics/stb.ixx
	${ROOT_PATH}/graphics/common_context.ixx
	${ROOT_PATH}/graphics/opengl/core.ixx
	${ROOT_PATH}/graphics/opengl/uniform_block.ixx
	${ROOT_PATH}/graphics/loco.ixx
	${ROOT_PATH}/graphics/graphics.ixx
	${ROOT_PATH}/graphics/file_dialog.ixx
	${ROOT_PATH}/graphics/algorithm/raycast_grid.ixx
	
	${ROOT_PATH}/physics/collision/rectangle.ixx
	${ROOT_PATH}/physics/collision/triangle.ixx
	
	${ROOT_PATH}/window/window.ixx
	${ROOT_PATH}/window/input_common.ixx
	${ROOT_PATH}/window/input.ixx
	
	${ROOT_PATH}/audio/audio.ixx
	
	${ROOT_PATH}/event/event.ixx
	
	${ROOT_PATH}/network/network.ixx
	
	${ROOT_PATH}/fan.ixx
	
	#external
	${ROOT_PATH}/video/screen_codec.ixx
)

if (FAN_JSON)
	list(APPEND MODULE_FILES
		${ROOT_PATH}/types/json.ixx
	)
endif()

if(FAN_FMT OR FAN_GUI)
	list(APPEND MODULE_FILES 
		${ROOT_PATH}/fmt.ixx
	)
endif()
if(FAN_GUI)
	list(APPEND MODULE_FILES
		${ROOT_PATH}/graphics/gui/tilemap_editor/renderer0.ixx
		${ROOT_PATH}/graphics/gui/tilemap_editor/loader.ixx
		${ROOT_PATH}/graphics/gui/tilemap_editor/editor.ixx
	
		${ROOT_PATH}/graphics/gui/gui.ixx
		${ROOT_PATH}/graphics/gui/console.ixx
	)
endif()
if(FAN_PHYSICS)
	list(APPEND MODULE_FILES
		${ROOT_PATH}/physics/b2_integration.ixx
		${ROOT_PATH}/graphics/physics_shapes.ixx
	)
endif()

target_sources(fan_modules
  PUBLIC
    FILE_SET CXX_MODULES
    BASE_DIRS ${ROOT_PATH}
    FILES ${MODULE_FILES}
)

set_target_properties(fan_modules PROPERTIES CXX_SCAN_FOR_MODULES OFF)
apply_common_compile_options(fan_modules)

# if (FAN_FMT)
	# add_library(fmt STATIC)
	# target_sources(fmt
  # PUBLIC
    # FILE_SET CXX_MODULES
    # BASE_DIRS ${ROOT_PATH}
    # FILES
			# ${ROOT_PATH}/fmt.ixx
	# )
	
	# apply_common_compile_options(fmt)
	# #target_link_libraries(fmt PRIVATE fan_pch)
	# set_target_properties(fmt PROPERTIES
		# ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
		# OUTPUT_NAME "fmt"
	# )
# endif()
	
if(FAN_GUI)
	set(IMGUI_SOURCES
		${ROOT_PATH}/imgui/imgui.cpp
		${ROOT_PATH}/imgui/imgui_draw.cpp
		${ROOT_PATH}/imgui/imgui_widgets.cpp
		${ROOT_PATH}/imgui/imgui_tables.cpp
		${ROOT_PATH}/imgui/imgui_impl_glfw.cpp
		${ROOT_PATH}/imgui/imgui_impl_opengl3.cpp
		${ROOT_PATH}/imgui/implot_items.cpp
		${ROOT_PATH}/imgui/implot.cpp
		${ROOT_PATH}/imgui/text_editor.cpp
	)
	if (FAN_VULKAN)
		list(APPEND IMGUI_SOURCES ${ROOT_PATH}/imgui/imgui_impl_vulkan.cpp)
	endif()
	
	add_library(imgui STATIC ${IMGUI_SOURCES})
	apply_common_compile_options(imgui)
	target_include_directories(imgui PRIVATE ${ROOT_PATH}/imgui)
	
	#target_link_libraries(imgui PRIVATE fan_pch)
	
	target_compile_definitions(imgui PRIVATE 
		IMGUI_IMPL_OPENGL_LOADER_CUSTOM
		IMGUI_DEFINE_MATH_OPERATORS
		IMGUI_DISABLE_SSE
		STBI_NO_SIMD
	)
	apply_fan_definitions(imgui)
	set_target_properties(imgui PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
		OUTPUT_NAME "imgui"
	)
endif()

if(UNIX AND FAN_GUI)
	set(NFD_SOURCES
		${ROOT_PATH}/nativefiledialog/nfd_common.c
		${ROOT_PATH}/nativefiledialog/nfd_gtk.c
	)
elseif(FAN_GUI)
	set(NFD_SOURCES
		${ROOT_PATH}/nativefiledialog/nfd_common.c
		${ROOT_PATH}/nativefiledialog/nfd_win.cpp
	)
endif()

if (FAN_GUI)
	add_library(nfd STATIC ${NFD_SOURCES})
	apply_common_compile_options(nfd)
	if(UNIX)
		target_include_directories(nfd PRIVATE ${GTK3_INCLUDE_DIRS})
	endif()
	set_target_properties(nfd PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
		OUTPUT_NAME "nfd"
	)
endif()


if(BUILD_MAIN)
	if(NOT DEFINED MAIN)
		set(MAIN examples/graphics/2D/shapes/rectangle.cpp)
	endif()
	
	add_executable(a.exe ${MAIN})
	add_dependencies(a.exe fan_modules)
	set_target_properties(a.exe PROPERTIES CXX_SCAN_FOR_MODULES ON)
	apply_common_compile_options(a.exe)
	target_link_libraries(a.exe PRIVATE fan_modules)
	apply_fan_definitions(a.exe)
	
	if(UNIX)
			target_link_libraries(a.exe PRIVATE
				webp
				glfw
				X11
				opus
				pulse-simple
				uv
				GL
				GLEW
				
				#stdc++exp
			)
			if (FAN_FMT)
				target_link_libraries(a.exe PRIVATE fmt)
			endif()
			if (FAN_GUI)
				target_link_libraries(a.exe PRIVATE PkgConfig::GTK3 nfd)
			endif()
			
			if(FAN_PHYSICS)
				if(NOT BOX2D_LIB)
					find_library(BOX2D_LIB box2d 
						PATHS
							${THIRDPARTY_PATH}/lib
							/usr/local/lib
							/usr/lib
						NO_DEFAULT_PATH
						NO_CMAKE_ENVIRONMENT_PATH
						NO_CMAKE_PATH
						NO_SYSTEM_ENVIRONMENT_PATH
						NO_CMAKE_SYSTEM_PATH
					)
					if(NOT BOX2D_LIB)
							message(FATAL_ERROR "Box2D not found")
					endif()
				endif()
				set(BOX2D_WITH_FLAGS "-Wl,--whole-archive ${BOX2D_LIB} -Wl,--no-whole-archive")
				target_link_libraries(a.exe PRIVATE ${BOX2D_WITH_FLAGS})
			endif()

			if(FAN_GUI)
				target_link_libraries(a.exe PRIVATE imgui)
			endif()
			
			if(FAN_3D)
				target_link_libraries(a.exe PRIVATE assimp)
			endif()
			
		if(FAN_VULKAN)
			if(NOT TARGET Vulkan::Vulkan)
				find_package(Vulkan REQUIRED)
			endif()
			
			if(NOT SHADERC_LIB)
				find_library(SHADERC_LIB shaderc_shared 
					PATHS 
						${THIRDPARTY_PATH}/lib
						/usr/local/lib
						/usr/lib
					NO_DEFAULT_PATH
					NO_CMAKE_ENVIRONMENT_PATH
					NO_CMAKE_PATH
					NO_SYSTEM_ENVIRONMENT_PATH
					NO_CMAKE_SYSTEM_PATH
				)
				if(NOT SHADERC_LIB)
					message(FATAL_ERROR "SHADERC not found")
				endif()
			endif()
			
			target_link_libraries(a.exe PRIVATE Vulkan::Vulkan ${SHADERC_LIB})
		endif()
	else()
		message(WARNING "Windows backend not implemented")
	endif()
endif()