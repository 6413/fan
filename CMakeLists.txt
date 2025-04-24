cmake_minimum_required(VERSION 3.20)
project(fan)

set(FAN_PHYSICS ON)
set(FAN_GUI ON)
set(FAN_JSON ON)
set(FAN_3D ON)
set(FAN_OPENGL ON)
set(FAN_VULKAN OFF)
set(FAN_FMT OFF)

set(BUILD_MAIN ON)


set(CMAKE_COLOR_DIAGNOSTICS ON)

set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (FAN_GUI)
	set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
endif()
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
if (FAN_GUI)
	set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
endif()
set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 2)


find_program(CCACHE_PROGRAM ccache NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
if(CCACHE_PROGRAM)
		set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

if (FAN_GUI)
	set(CMAKE_C_COMPILER "/usr/bin/clang")
endif()
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=lld")

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

function(apply_fan_definitions target)
	target_compile_definitions(${target} PRIVATE
		$<$<BOOL:${FAN_OPENGL}>:fan_opengl>
		$<$<BOOL:${FAN_GUI}>:fan_gui>
		$<$<BOOL:${FAN_JSON}>:fan_json>
		$<$<BOOL:${FAN_3D}>:fan_3d>
		$<$<BOOL:${FAN_PHYSICS}>:fan_physics>
		$<$<BOOL:${FAN_VULKAN}>:fan_vulkan>
		IMGUI_IMPL_OPENGL_LOADER_CUSTOM
		IMGUI_DEFINE_MATH_OPERATORS
	)
endfunction()

function(apply_common_compile_options target)
	target_compile_options(${target} PRIVATE
		-w -MD -MT -Wall -Wextra -ferror-limit=4 -stdlib=libstdc++ -fsized-deallocation -fPIC -ftime-trace
		-Wno-shift-op-parentheses -Wno-unused-variable -Wno-int-to-void-pointer-cast
		-Wno-unused-parameter -Wno-unused-function -Wno-bitwise-op-parentheses
		-Wno-invalid-offsetof -Wno-missing-field-initializers -Wno-sign-compare
		-Wno-unused-but-set-parameter -Wno-unused-value
	)
endfunction()

add_library(fan_pch INTERFACE)
target_precompile_headers(fan_pch INTERFACE ${ROOT_PATH}/pch.h)

if (FAN_FMT)
	add_library(fmt STATIC ${ROOT_PATH}/fmt.cpp)
	apply_common_compile_options(fmt)
	target_link_libraries(fmt PRIVATE fan_pch)
	set_target_properties(fmt PROPERTIES
		ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
		OUTPUT_NAME "fmt"
	)
endif()
	
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
	target_link_libraries(imgui PRIVATE fan_pch)
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

set(FAN_GRAPHICS_SOURCES "")
set(FAN_PHYSICS_SOURCES "")
set(FAN_VULKAN_SOURCES "")

set(FAN_GRAPHICS_SOURCES 
	${ROOT_PATH}/graphics/algorithm/AStar.cpp
	${ROOT_PATH}/graphics/camera.cpp
	${ROOT_PATH}/graphics/console_impl.cpp
	${ROOT_PATH}/graphics/graphics.cpp
	${ROOT_PATH}/graphics/gui/gui.cpp
	${ROOT_PATH}/graphics/loco.cpp
	${ROOT_PATH}/graphics/opengl/gl_core.cpp
	${ROOT_PATH}/graphics/opengl/uniform_block.cpp	
	${ROOT_PATH}/types/color.cpp
	${ROOT_PATH}/types/fstring.cpp
	${ROOT_PATH}/types/types.cpp
	${ROOT_PATH}/window/window.cpp
	${ROOT_PATH}/window/window_input_common.cpp
	
	${ROOT_PATH}/io/file.cpp
)

if(FAN_VULKAN)
	list(APPEND FAN_GRAPHICS_SOURCES 
		${ROOT_PATH}/graphics/vulkan/vk_core.cpp
	)
endif()

set(FAN_SOURCES ${FAN_GRAPHICS_SOURCES})

if(FAN_PHYSICS)
	set (FAN_PHYSICS_SOURCES 
		${ROOT_PATH}/physics/b2_integration.cpp
		${ROOT_PATH}/graphics/physics_shapes.cpp
	)
	list(APPEND FAN_SOURCES ${FAN_PHYSICS_SOURCES})
endif()

add_library(fan_obj OBJECT ${FAN_SOURCES})
apply_common_compile_options(fan_obj)
target_link_libraries(fan_obj PRIVATE fan_pch)
apply_fan_definitions(fan_obj)

add_library(fan SHARED $<TARGET_OBJECTS:fan_obj>)
apply_common_compile_options(fan)
apply_fan_definitions(fan)
target_link_libraries(fan PRIVATE fan_pch)
set_target_properties(fan PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_PATH}
	OUTPUT_NAME "fan"
)

if(BUILD_MAIN)
	if(NOT DEFINED MAIN)
		set(MAIN examples/graphics/2D/shapes/rectangle.cpp)
	endif()
	
	add_executable(a.exe ${MAIN} $<TARGET_OBJECTS:fan_obj>)
	apply_common_compile_options(a.exe)
	target_link_libraries(a.exe PRIVATE fan_pch)
	apply_fan_definitions(a.exe)
	
	if(UNIX)
			target_link_libraries(a.exe PRIVATE
				fan
				webp
				glfw
				X11
				opus
				pulse-simple
				uv
				GL
				GLEW
				stdc++exp
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
				target_link_libraries(fan PRIVATE ${BOX2D_WITH_FLAGS})
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