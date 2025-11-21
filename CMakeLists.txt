cmake_minimum_required(VERSION 3.31.1)

set(FAN_PHYSICS ON)
set(FAN_GUI ON)
set(FAN_JSON ON)
set(FAN_3D OFF)
set(FAN_OPENGL ON)
set(FAN_VULKAN OFF)
set(FAN_FMT ON)
set(FAN_WAYLAND_SCREEN OFF)
set(FAN_NETWORK OFF)

set(BUILD_MAIN ON)
set(BUILD_FAN_LIBRARY OFF CACHE STRING "Build fan as library: OFF, STATIC, or SHARED")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
set(CMAKE_CXX_MODULE_STD ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)
set(CMAKE_UNITY_BUILD OFF)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 2)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

set(STDLIB_FLAGS "-stdlib=libc++ -lc++abi")

if(WIN32)
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()

  set(CMAKE_EXECUTABLE_SUFFIX_CXX "")
  set(CMAKE_C_COMPILER "clang")
  set(CMAKE_CXX_COMPILER "clang++")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -MT -D_DEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -ffunction-sections -fdata-sections -MT -DNDEBUG")
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,/FORCE:MULTIPLE -Wl,--gc-sections ")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Wl,/FORCE:MULTIPLE -Wl,--gc-sections -s")
  set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "clang-scan-deps")
else()
  find_program(CCACHE_PROGRAM ccache NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()

  set(CMAKE_C_COMPILER "/usr/bin/clang-20")
  set(CMAKE_CXX_COMPILER "/usr/bin/clang++-20")
  set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4 -D_DEBUG=4 -pthread")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-Wl,--strip-all")
  set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-s")
  set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=gold")
  set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=gold")
  set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "/usr/bin/clang-scan-deps-20")
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_CXX_MODULE_MAP_FLAG "-fmodule-mapper=")
set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "-fmodule-only")
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT ${STDLIB_FLAGS})
set(CMAKE_SHARED_LINKER_FLAGS_INIT ${STDLIB_FLAGS})
set(CMAKE_MODULE_LINKER_FLAGS_INIT ${STDLIB_FLAGS})

project(fan)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${STDLIB_FLAGS}")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ROOT_PATH ${SOURCE_DIR}/fan/)
set(OUTPUT_PATH ${SOURCE_DIR}/lib/fan)
set(THIRDPARTY_PATH ${SOURCE_DIR}/thirdparty/fan)

file(MAKE_DIRECTORY ${OUTPUT_PATH})

include_directories(${SOURCE_DIR})

if(WIN32)
  include_directories(${THIRDPARTY_PATH}/include ${SOURCE_DIR}/include)
  if(FAN_3D)
    include_directories(a.exe PRIVATE C:/Program\ Files/Assimp/include)
  endif()
  link_directories(${SOURCE_DIR}/lib)
else()
  include_directories(${THIRDPARTY_PATH}/include)
  link_directories(${THIRDPARTY_PATH}/lib)
endif()

function(setup_library lib_name lib_path include_path)
  find_library(${lib_name}_LIBRARY ${lib_name} PATHS ${lib_path} NO_DEFAULT_PATH)
  if(NOT ${lib_name}_LIBRARY)
    if(ARGC GREATER 3)
      message(${ARGV3} "${lib_name} library not found")
      set(${lib_name}_FOUND FALSE PARENT_SCOPE)
      return()
    else()
      message(FATAL_ERROR "${lib_name} not found")
    endif()
  endif()
  add_library(${lib_name} STATIC IMPORTED)
  set_target_properties(${lib_name} PROPERTIES
    IMPORTED_LOCATION ${${lib_name}_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${include_path}
  )
  set(${lib_name}_FOUND TRUE PARENT_SCOPE)
  message(STATUS "Using ${lib_name} from ${${lib_name}_LIBRARY}")
endfunction()

if(FAN_GUI)
  if(WIN32)
    setup_library(freetype ${SOURCE_DIR}/lib/freetype ${SOURCE_DIR}/include)
    set(Freetype_VERSION "2.13.2")
    add_library(Freetype::Freetype ALIAS freetype)
    message(STATUS "Using Windows FreeType ${Freetype_VERSION}")

    setup_library(lunasvg ${SOURCE_DIR}/lib ${SOURCE_DIR}/include WARNING)
    if(NOT lunasvg_FOUND)
      message(WARNING "LunaSVG library not found - SVG glyph support disabled")
      set(LUNASVG_FOUND FALSE)
    else()
      set(LUNASVG_FOUND TRUE)
    endif()

    add_compile_definitions(__HAVE_WIN32_GUI=1)
  else()
    set(CMAKE_PREFIX_PATH "${THIRDPARTY_PATH};${CMAKE_PREFIX_PATH}")
    find_package(Freetype REQUIRED)

    if(NOT Freetype_VERSION OR Freetype_VERSION STREQUAL "")
      find_file(FREETYPE_H_FILE freetype.h
        PATHS ${FREETYPE_INCLUDE_DIRS}
        PATH_SUFFIXES freetype freetype2/freetype
        NO_DEFAULT_PATH
      )

      if(FREETYPE_H_FILE)
        file(READ "${FREETYPE_H_FILE}" FREETYPE_H_CONTENT)
        string(REGEX MATCH "#define[ \t]+FREETYPE_MAJOR[ \t]+([0-9]+)" _ "${FREETYPE_H_CONTENT}")
        set(FREETYPE_VERSION_MAJOR ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define[ \t]+FREETYPE_MINOR[ \t]+([0-9]+)" _ "${FREETYPE_H_CONTENT}")
        set(FREETYPE_VERSION_MINOR ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define[ \t]+FREETYPE_PATCH[ \t]+([0-9]+)" _ "${FREETYPE_H_CONTENT}")
        set(FREETYPE_VERSION_PATCH ${CMAKE_MATCH_1})

        if(FREETYPE_VERSION_MAJOR AND FREETYPE_VERSION_MINOR AND FREETYPE_VERSION_PATCH)
          set(Freetype_VERSION "${FREETYPE_VERSION_MAJOR}.${FREETYPE_VERSION_MINOR}.${FREETYPE_VERSION_PATCH}")
        else()
          set(Freetype_VERSION "2.13.2")
        endif()
      else()
        set(Freetype_VERSION "2.13.2")
      endif()
    endif()

    if(Freetype_VERSION VERSION_LESS "2.12")
      message(FATAL_ERROR "FreeType version ${Freetype_VERSION} found, but version 2.12+ is required for SVG glyph support. Please run: ./install.sh")
    endif()

    message(STATUS "Using FreeType ${Freetype_VERSION}")

    setup_library(lunasvg ${THIRDPARTY_PATH}/lib ${THIRDPARTY_PATH}/include)
    set(LUNASVG_FOUND TRUE)

    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH OFF)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)
    if(FAN_WAYLAND_SCREEN)
      pkg_check_modules(WAYLAND_CLIENT wayland-client)
      pkg_check_modules(DBUS REQUIRED dbus-1)
      pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
      include_directories(${DBUS_INCLUDE_DIRS} ${PIPEWIRE_INCLUDE_DIRS} ${WAYLAND_CLIENT_INCLUDE_DIRS})
      link_directories(${DBUS_LIBRARY_DIRS} ${PIPEWIRE_LIBRARY_DIRS} ${WAYLAND_CLIENT_LIBRARY_DIRS})
      add_compile_definitions(__HAVE_WAYLAND=1)
    endif()
  endif()
endif()

if(BUILD_FAN_LIBRARY STREQUAL "STATIC" OR BUILD_FAN_LIBRARY STREQUAL "SHARED")
  set(BUILD_SHARED_LIBS OFF)
else()
  set(BUILD_SHARED_LIBS ON)
endif()

function(apply_imgui_definitions target)
  target_compile_definitions(${target} PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_CUSTOM IMGUI_DEFINE_MATH_OPERATORS
    IMGUI_DISABLE_SSE IMGUI_ENABLE_FREETYPE
  )
endfunction()

function(apply_fan_definitions target)
  apply_imgui_definitions(${target})
  target_compile_definitions(${target} PRIVATE
    $<$<BOOL:${FAN_OPENGL}>:fan_opengl>
    $<$<BOOL:${FAN_GUI}>:fan_gui>
    $<$<BOOL:${FAN_JSON}>:fan_json>
    $<$<BOOL:${FAN_3D}>:fan_3D>
    $<$<BOOL:${FAN_PHYSICS}>:fan_physics>
    $<$<BOOL:${FAN_VULKAN}>:fan_vulkan>
		$<$<BOOL:${FAN_NETWORK}>:fan_network>
    loco_line loco_rectangle loco_sprite loco_light loco_circle
    loco_responsive_text loco_universal_image_renderer
  )
endfunction()

function(apply_common_compile_options target)
  set(COMMON_WARNINGS -Wall -Wextra -ferror-limit=20 -Wno-shift-op-parentheses
    -Wno-unused-variable -Wno-int-to-void-pointer-cast -Wno-unused-parameter
    -Wno-unused-function -Wno-bitwise-op-parentheses -Wno-invalid-offsetof
    -Wno-missing-field-initializers -Wno-sign-compare -Wno-unused-but-set-parameter
    -Wno-unused-value -w)

  if(WIN32)
    target_compile_options(${target} PRIVATE ${COMMON_WARNINGS}
      -D_ITERATOR_DEBUG_LEVEL=0 -DGLEW_STATIC -DSTBI_NO_SIMD)
  else()
    target_compile_options(${target} PRIVATE ${COMMON_WARNINGS}
      -MT -fsized-deallocation -DSTBI_NO_SIMD -fmacro-backtrace-limit=0)
  endif()
endfunction()

if(BUILD_FAN_LIBRARY STREQUAL "STATIC")
  add_library(fan_modules STATIC)
  message(STATUS "Building fan as STATIC library")
elseif(BUILD_FAN_LIBRARY STREQUAL "SHARED")
  add_library(fan_modules SHARED)
  message(STATUS "Building fan as SHARED library")
else()
  add_library(fan_modules STATIC)
  message(STATUS "Building fan for executable only (internal static)")
endif()

set_target_properties(fan_modules PROPERTIES
	CXX_MODULE_DIRECTORY ${OUTPUT_PATH}
	UNITY_BUILD OFF
)

apply_fan_definitions(fan_modules)

set(MODULE_FILES
  ${ROOT_PATH}/types/types.ixx
  ${ROOT_PATH}/types/color.ixx
  ${ROOT_PATH}/types/vector.ixx
  ${ROOT_PATH}/types/quaternion.ixx
  ${ROOT_PATH}/types/matrix.ixx
  ${ROOT_PATH}/types/magic.ixx
  ${ROOT_PATH}/types/masterpiece.ixx
  ${ROOT_PATH}/types/fstring.ixx
  ${ROOT_PATH}/math/math.ixx
  ${ROOT_PATH}/time.ixx
  ${ROOT_PATH}/utility.ixx
  ${ROOT_PATH}/print.ixx
  ${ROOT_PATH}/random.ixx
  ${ROOT_PATH}/io/directory.ixx
  ${ROOT_PATH}/io/file.ixx
  ${ROOT_PATH}/graphics/common_types.ixx
  ${ROOT_PATH}/graphics/camera.ixx
  ${ROOT_PATH}/graphics/image_load.ixx
  ${ROOT_PATH}/graphics/webp.ixx
  ${ROOT_PATH}/graphics/stb.ixx
  ${ROOT_PATH}/graphics/common_context.ixx
  ${ROOT_PATH}/graphics/opengl/gl_core.ixx
  ${ROOT_PATH}/graphics/opengl/uniform_block.ixx
	${ROOT_PATH}/texture_pack/tp0.ixx
	${ROOT_PATH}/graphics/shapes.ixx
  ${ROOT_PATH}/graphics/loco.ixx
  ${ROOT_PATH}/graphics/graphics.ixx
  ${ROOT_PATH}/graphics/file_dialog.ixx
  ${ROOT_PATH}/graphics/algorithm/raycast_grid.ixx
	${ROOT_PATH}/graphics/algorithm/pathfind.ixx
  ${ROOT_PATH}/physics/collision/rectangle.ixx
  ${ROOT_PATH}/physics/collision/triangle.ixx
	${ROOT_PATH}/physics/collision/circle.ixx
	${ROOT_PATH}/physics/physics_types.ixx
  ${ROOT_PATH}/window/window.ixx
  ${ROOT_PATH}/window/input_common.ixx
  ${ROOT_PATH}/window/input.ixx
	${ROOT_PATH}/window/input_action.ixx
  ${ROOT_PATH}/audio/audio.ixx
	${ROOT_PATH}/event/event_types.ixx
  ${ROOT_PATH}/event/event.ixx
  ${ROOT_PATH}/network/network.ixx
  ${ROOT_PATH}/graphics/graphics_network.ixx
	${ROOT_PATH}/noise.ixx
  ${ROOT_PATH}/fan.ixx
)

if(FAN_WAYLAND_SCREEN)
  list(APPEND MODULE_FILES ${ROOT_PATH}/video/screen_codec.ixx)
endif()

if(FAN_JSON)
  list(APPEND MODULE_FILES ${ROOT_PATH}/types/json.ixx)
endif()

if(FAN_FMT OR FAN_GUI)
  list(APPEND MODULE_FILES ${ROOT_PATH}/fmt.ixx)
endif()

if(FAN_GUI)
  list(APPEND MODULE_FILES
    ${ROOT_PATH}/graphics/gui/text_logger.ixx
		${ROOT_PATH}/graphics/gui/gui_types.ixx
    ${ROOT_PATH}/graphics/gui/gui.ixx
    ${ROOT_PATH}/graphics/gui/console.ixx
		${ROOT_PATH}/graphics/gui/tilemap_editor/loader.ixx
		${ROOT_PATH}/graphics/gui/tilemap_editor/renderer0.ixx
  )
endif()

if(FAN_PHYSICS)
  list(APPEND MODULE_FILES
    ${ROOT_PATH}/physics/b2_integration.ixx
    ${ROOT_PATH}/physics/physics_common_context.ixx
		${ROOT_PATH}/graphics/physics_shapes.ixx
  )
endif()

if(FAN_3D)
	list(APPEND MODULE_FILES
    ${ROOT_PATH}/graphics/opengl/3D/objects/fms.ixx
    ${ROOT_PATH}/graphics/opengl/3D/objects/model.ixx
  )
endif()

foreach(m ${MODULE_FILES})
  get_filename_component(MDIR ${m} DIRECTORY)
  get_filename_component(MNAME ${m} NAME_WE)
  set(IMPL_FILE "${MDIR}/${MNAME}_impl.cpp")
  if(EXISTS "${IMPL_FILE}")
    list(APPEND IMPL_SOURCES "${IMPL_FILE}")
    set_source_files_properties("${IMPL_FILE}" PROPERTIES UNITY_BUILD OFF)
  endif()
endforeach()

message(STATUS "Found impl sources: ${IMPL_SOURCES}")

target_sources(fan_modules
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS ${ROOT_PATH} FILES ${MODULE_FILES}
  PRIVATE ${IMPL_SOURCES} ${ROOT_PATH}/graphics/algorithm/AStar.cpp
)

set_target_properties(fan_modules PROPERTIES CXX_SCAN_FOR_MODULES ON)
apply_common_compile_options(fan_modules)

target_compile_options(fan_modules PRIVATE
  -fprebuilt-module-path=${CMAKE_BINARY_DIR}/CMakeFiles/fan_modules.dir
)

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
    ${ROOT_PATH}/imgui/misc/freetype/imgui_freetype.cpp
  )
  if(FAN_VULKAN)
    list(APPEND IMGUI_SOURCES ${ROOT_PATH}/imgui/imgui_impl_vulkan.cpp)
  endif()
  if(FAN_3D AND FAN_GUI)
    list(APPEND IMGUI_SOURCES ${ROOT_PATH}/imgui/ImGuizmo.cpp)
  endif()

  add_library(imgui STATIC ${IMGUI_SOURCES})
	
  apply_common_compile_options(imgui)
  set_target_properties(imgui PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    UNITY_BUILD OFF
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "imgui"
		CXX_SCAN_FOR_MODULES OFF
  )
  target_include_directories(imgui PRIVATE
    ${ROOT_PATH}/imgui
    ${ROOT_PATH}/imgui/misc/freetype
    ${FREETYPE_INCLUDE_DIRS}
  )
  target_link_libraries(imgui PRIVATE Freetype::Freetype)
  if(LUNASVG_FOUND)
    target_link_libraries(imgui PRIVATE lunasvg)
    target_compile_definitions(imgui PRIVATE IMGUI_ENABLE_FREETYPE_LUNASVG)
  endif()
  apply_imgui_definitions(imgui)
  apply_fan_definitions(imgui)

  if(UNIX)
    set(NFD_SOURCES
      ${ROOT_PATH}/nativefiledialog/nfd_common.c
      ${ROOT_PATH}/nativefiledialog/nfd_gtk.c
    )
  else()
    set(NFD_SOURCES
      ${ROOT_PATH}/nativefiledialog/nfd_common.c
      ${ROOT_PATH}/nativefiledialog/nfd_win.cpp
    )
  endif()

  add_library(nfd STATIC ${NFD_SOURCES})
  apply_common_compile_options(nfd)
  set_target_properties(nfd PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "nfd"
  )
  if(UNIX)
    target_include_directories(nfd PRIVATE ${GTK3_INCLUDE_DIRS})
  endif()
endif()

function(link_common_libs target visibility)
  if(UNIX)
    target_link_libraries(${target} ${visibility}
      webp glfw X11 opus pulse-simple uv GL GLEW ssl crypto png z curl
    )
    if(FAN_WAYLAND_SCREEN)
      target_link_libraries(${target} ${visibility}
        ${WAYLAND_CLIENT_LIBRARIES} ${PIPEWIRE_LIBRARIES} ${DBUS_LIBRARIES}
        avcodec avutil swscale
      )
    endif()
    if(FAN_FMT)
      target_link_libraries(${target} ${visibility} fmt)
    endif()
    if(FAN_GUI)
      if(BUILD_FAN_LIBRARY STREQUAL "SHARED" AND visibility STREQUAL "PRIVATE")
        target_link_libraries(${target} ${visibility} PkgConfig::GTK3)
      else()
        target_link_libraries(${target} ${visibility}
          PkgConfig::GTK3 nfd imgui Freetype::Freetype
        )
        if(LUNASVG_FOUND)
          target_link_libraries(${target} ${visibility} lunasvg)
        endif()
      endif()
    endif()
    if(FAN_PHYSICS)
      find_library(BOX2D_LIB box2d PATHS ${THIRDPARTY_PATH}/lib NO_DEFAULT_PATH)
      if(NOT BOX2D_LIB)
        message(FATAL_ERROR "Box2D not found")
      endif()
      if(BUILD_FAN_LIBRARY STREQUAL "SHARED" AND visibility STREQUAL "PRIVATE")
        target_link_options(${target} ${visibility}
          "LINKER:--whole-archive,${BOX2D_LIB},--no-whole-archive"
        )
      else()
        target_link_libraries(${target} ${visibility}
          -Wl,--whole-archive ${BOX2D_LIB} -Wl,--no-whole-archive
        )
      endif()
    endif()
    if(FAN_3D)
      target_link_libraries(${target} ${visibility} assimp)
    endif()
    if(FAN_VULKAN)
      find_package(Vulkan REQUIRED)
      find_library(SHADERC_LIB shaderc_shared PATHS ${THIRDPARTY_PATH}/lib NO_DEFAULT_PATH)
      if(NOT SHADERC_LIB)
        message(FATAL_ERROR "SHADERC not found")
      endif()
      target_link_libraries(${target} ${visibility} Vulkan::Vulkan ${SHADERC_LIB})
    endif()
  else()
    target_link_libraries(${target} ${visibility}
      opengl32
      ${SOURCE_DIR}/lib/GLFW/glfw3_mt.lib
      ${SOURCE_DIR}/lib/GLEW/glew32s.lib
      ${SOURCE_DIR}/lib/libuv/uv_a.lib
      ${SOURCE_DIR}/lib/libwebp/libwebp.lib
      ${SOURCE_DIR}/lib/opus/opus.lib
      ${SOURCE_DIR}/lib/openssl/libssl.lib
      ${SOURCE_DIR}/lib/openssl/libcrypto.lib
    )
    if(FAN_GUI)
      target_link_libraries(${target} ${visibility}
        imgui nfd
        ${SOURCE_DIR}/lib/freetype/freetype.lib
        ${SOURCE_DIR}/lib/lunasvg/lunasvg.lib
      )
    endif()
    if(FAN_PHYSICS)
      find_library(BOX2D_LIB box2d PATHS ${SOURCE_DIR}/lib/box2d NO_DEFAULT_PATH)
      if(BOX2D_LIB)
        target_link_libraries(${target} ${visibility} ${BOX2D_LIB})
      else()
        message(WARNING "Box2D not found for Windows")
      endif()
    endif()
    if(FAN_3D)
      target_link_libraries(${target} ${visibility}
        C:/Program\ Files/Assimp/lib/x64/assimp-vc143-mt.lib
      )
    endif()
    if(FAN_VULKAN)
      find_package(Vulkan REQUIRED)
      find_library(SHADERC_LIB shaderc_shared PATHS ${SOURCE_DIR}/lib NO_DEFAULT_PATH)
      if(SHADERC_LIB)
        target_link_libraries(${target} ${visibility} Vulkan::Vulkan ${SHADERC_LIB})
      else()
        message(WARNING "SHADERC not found for Windows")
      endif()
    endif()
    if(FAN_WAYLAND_SCREEN)
      set(OPENH264_LIBS
        ${SOURCE_DIR}/lib/openh264/welsdcore.lib
        ${SOURCE_DIR}/lib/openh264/welsecore.lib
        ${SOURCE_DIR}/lib/openh264/WelsDecPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsEncPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsVP.lib
      )
      target_link_libraries(${target} ${visibility}
        DXGI D3D11
        ${SOURCE_DIR}/lib/libx264/libx264.lib
        ${OPENH264_LIBS}
      )
    endif()
  endif()
endfunction()

if(BUILD_FAN_LIBRARY STREQUAL "STATIC" OR BUILD_FAN_LIBRARY STREQUAL "SHARED")
  link_common_libs(fan_modules PUBLIC)
  set_target_properties(fan_modules PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "fan"
  )
endif()

if(BUILD_MAIN)

  if(NOT DEFINED MAIN)
    set(MAIN ${SOURCE_DIR}/examples/engine_demos/engine_demo.cpp)
  endif()

  if(UNIX AND WAYLAND_CLIENT_FOUND AND FAN_WAYLAND_SCREEN)
    add_executable(a.exe
      ${ROOT_PATH}/video/xdg-output-unstable-v1-client-protocol.c
      ${ROOT_PATH}/video/wlr-screencopy-unstable-v1-protocol.c
      ${MAIN}
    )
    target_link_libraries(a.exe PRIVATE ${WAYLAND_CLIENT_LIBRARIES})
  else()
    add_executable(a.exe ${MAIN})
    target_compile_definitions(a.exe PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  endif()

	# target_sources(a.exe PRIVATE
		# ${ROOT_PATH}/graphics/algorithm/AStar.cpp
	# )

  if(DEFINED CUSTOM_LIBS)
    target_link_libraries(fan_modules PRIVATE ${CUSTOM_LIBS})
    target_link_libraries(a.exe PRIVATE ${CUSTOM_LIBS})
  endif()

  add_dependencies(a.exe fan_modules)
  set_target_properties(a.exe PROPERTIES CXX_SCAN_FOR_MODULES ON)
  apply_common_compile_options(a.exe)
  target_link_libraries(a.exe PRIVATE fan_modules)
  apply_fan_definitions(a.exe)
  target_compile_options(a.exe PRIVATE
    -fprebuilt-module-path=${CMAKE_BINARY_DIR}/CMakeFiles/fan_modules.dir
  )

  if(BUILD_FAN_LIBRARY STREQUAL "OFF" OR NOT BUILD_FAN_LIBRARY)
    link_common_libs(a.exe PRIVATE)
  elseif(BUILD_FAN_LIBRARY STREQUAL "SHARED")
    if(UNIX AND FAN_PHYSICS)
      find_library(BOX2D_LIB box2d PATHS ${THIRDPARTY_PATH}/lib NO_DEFAULT_PATH)
      if(BOX2D_LIB)
        target_link_libraries(a.exe PRIVATE
          -Wl,--whole-archive ${BOX2D_LIB} -Wl,--no-whole-archive
        )
      endif()
    endif()
    if(WIN32 AND FAN_WAYLAND_SCREEN)
      set(OPENH264_LIBS
        ${SOURCE_DIR}/lib/openh264/welsdcore.lib
        ${SOURCE_DIR}/lib/openh264/welsecore.lib
        ${SOURCE_DIR}/lib/openh264/WelsDecPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsEncPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsVP.lib
      )
      target_link_libraries(a.exe PRIVATE
        DXGI.lib D3D11.lib
        ${SOURCE_DIR}/lib/libx264/libx264.lib
        ${OPENH264_LIBS}
        ${SOURCE_DIR}/lib/openh264/encConsole.lib
      )
    endif()
  endif()
endif()