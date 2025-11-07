cmake_minimum_required(VERSION 3.31.1)

# Todo add fan_network

set(FAN_PHYSICS ON)
set(FAN_GUI ON)
set(FAN_JSON ON)
set(FAN_3D OFF)
set(FAN_OPENGL ON)
set(FAN_VULKAN OFF)
set(FAN_FMT ON)
set(FAN_WAYLAND_SCREEN ON)

set(BUILD_MAIN ON)
set(BUILD_FAN_LIBRARY OFF CACHE STRING "Build fan as library: OFF, STATIC, or SHARED")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 2)

if(WIN32)
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
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
  set(CMAKE_CXX_MODULE_MAP_FLAG "-fmodule-mapper=")
  set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "-fmodule-only")
else()
  find_program(CCACHE_PROGRAM ccache NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()

  set(CMAKE_C_COMPILER "/usr/bin/clang")
  set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
  set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4 -D_DEBUG=4")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-Wl,--strip-all")
  set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-s")
  set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=gold")
  set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=gold")
  set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "/usr/bin/clang-scan-deps")
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
  set(CMAKE_CXX_MODULE_MAP_FLAG "-fmodule-mapper=")
  set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "-fmodule-only")
endif()

project(fan)

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

if(FAN_GUI)
  if(WIN32)
    find_library(FREETYPE_LIBRARY freetype PATHS ${SOURCE_DIR}/lib/freetype NO_DEFAULT_PATH)
    if(NOT FREETYPE_LIBRARY)
      message(FATAL_ERROR "FreeType library not found in lib/freetype/")
    endif()

    set(Freetype_VERSION "2.13.2")
    set(FREETYPE_INCLUDE_DIRS ${SOURCE_DIR}/include)

    add_library(Freetype::Freetype STATIC IMPORTED)
    set_target_properties(Freetype::Freetype PROPERTIES
      IMPORTED_LOCATION ${FREETYPE_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${FREETYPE_INCLUDE_DIRS}
    )

    message(STATUS "Using Windows FreeType ${Freetype_VERSION} from ${FREETYPE_LIBRARY}")

    find_library(LUNASVG_LIBRARY lunasvg PATHS ${SOURCE_DIR}/lib NO_DEFAULT_PATH)
    if(NOT LUNASVG_LIBRARY)
      message(WARNING "LunaSVG library not found - SVG glyph support disabled")
      set(LUNASVG_FOUND FALSE)
    else()
      add_library(lunasvg STATIC IMPORTED)
      set_target_properties(lunasvg PROPERTIES
        IMPORTED_LOCATION ${LUNASVG_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${SOURCE_DIR}/include
      )
      set(LUNASVG_FOUND TRUE)
      message(STATUS "Using Windows LunaSVG from ${LUNASVG_LIBRARY}")
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

    find_library(LUNASVG_LIBRARY lunasvg PATHS ${THIRDPARTY_PATH}/lib NO_DEFAULT_PATH)
    if(NOT LUNASVG_LIBRARY)
      message(FATAL_ERROR "LunaSVG not found. Please run: ./install.sh")
    endif()

    add_library(lunasvg STATIC IMPORTED)
    set_target_properties(lunasvg PROPERTIES
      IMPORTED_LOCATION ${LUNASVG_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${THIRDPARTY_PATH}/include
    )
    set(LUNASVG_FOUND TRUE)

    message(STATUS "Using LunaSVG from ${LUNASVG_LIBRARY}")

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

function(apply_fan_definitions target)
  target_compile_definitions(${target} PRIVATE
    $<$<BOOL:${FAN_OPENGL}>:fan_opengl>
    $<$<BOOL:${FAN_GUI}>:fan_gui>
    $<$<BOOL:${FAN_JSON}>:fan_json>
    $<$<BOOL:${FAN_3D}>:fan_3D>
    $<$<BOOL:${FAN_PHYSICS}>:fan_physics>
    $<$<BOOL:${FAN_VULKAN}>:fan_vulkan>
    loco_line loco_rectangle loco_sprite loco_light loco_circle
    loco_responsive_text loco_universal_image_renderer
    IMGUI_IMPL_OPENGL_LOADER_CUSTOM IMGUI_DEFINE_MATH_OPERATORS
    IMGUI_DISABLE_SSE IMGUI_ENABLE_FREETYPE
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
      -MT -stdlib=libstdc++ -fsized-deallocation -pthread -DSTBI_NO_SIMD
      -fmacro-backtrace-limit=0)
  endif()
endfunction()

if(BUILD_FAN_LIBRARY STREQUAL "STATIC")
  add_library(fan_modules STATIC)
	set_target_properties(fan_modules PROPERTIES
		CXX_MODULE_DIRECTORY ${OUTPUT_PATH}
	)
  message(STATUS "Building fan as STATIC library")
elseif(BUILD_FAN_LIBRARY STREQUAL "SHARED")
  add_library(fan_modules SHARED)
	set_target_properties(fan_modules PROPERTIES
		CXX_MODULE_DIRECTORY ${OUTPUT_PATH}
	)
  message(STATUS "Building fan as SHARED library")
else()
  add_library(fan_modules STATIC)
  message(STATUS "Building fan for executable only (internal static)")
endif()

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
  ${ROOT_PATH}/physics/collision/rectangle.ixx
  ${ROOT_PATH}/physics/collision/triangle.ixx
  ${ROOT_PATH}/window/window.ixx
  ${ROOT_PATH}/window/input_common.ixx
  ${ROOT_PATH}/window/input.ixx
	${ROOT_PATH}/window/input_action.ixx
  ${ROOT_PATH}/audio/audio.ixx
  ${ROOT_PATH}/event/event.ixx
  ${ROOT_PATH}/network/network.ixx
  ${ROOT_PATH}/graphics/graphics_network.ixx
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
    ${ROOT_PATH}/graphics/gui/gui.ixx
    ${ROOT_PATH}/graphics/gui/console.ixx
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

target_sources(fan_modules
  PUBLIC FILE_SET CXX_MODULES BASE_DIRS ${ROOT_PATH} FILES ${MODULE_FILES}
)

set_target_properties(fan_modules PROPERTIES CXX_SCAN_FOR_MODULES OFF)
apply_common_compile_options(fan_modules)

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
  target_compile_definitions(imgui PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_CUSTOM IMGUI_DEFINE_MATH_OPERATORS
    IMGUI_DISABLE_SSE IMGUI_ENABLE_FREETYPE
  )
  apply_fan_definitions(imgui)
endif()

if(FAN_GUI)
  if(UNIX)
    set(NFD_SOURCES
      ${ROOT_PATH}/nativefiledialog/nfd_common.c
      ${ROOT_PATH}/nativefiledialog/nfd_gtk.c
    )
  elseif(WIN32)
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
  elseif(WIN32)
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
      target_link_libraries(${target} ${visibility}
        DXGI D3D11
        ${SOURCE_DIR}/lib/libx264/libx264.lib
        ${SOURCE_DIR}/lib/openh264/welsdcore.lib
        ${SOURCE_DIR}/lib/openh264/welsecore.lib
        ${SOURCE_DIR}/lib/openh264/WelsDecPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsEncPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsVP.lib
      )
    endif()
  endif()
endfunction()

if(BUILD_FAN_LIBRARY STREQUAL "STATIC" OR BUILD_FAN_LIBRARY STREQUAL "SHARED")
  if(BUILD_FAN_LIBRARY STREQUAL "SHARED")
    link_common_libs(fan_modules PUBLIC)
  else()
    link_common_libs(fan_modules PUBLIC)
  endif()

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
      target_link_libraries(a.exe PRIVATE
        DXGI.lib D3D11.lib
        ${SOURCE_DIR}/lib/libx264/libx264.lib
        ${SOURCE_DIR}/lib/openh264/welsdcore.lib
        ${SOURCE_DIR}/lib/openh264/welsecore.lib
        ${SOURCE_DIR}/lib/openh264/WelsDecPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsEncPlus.lib
        ${SOURCE_DIR}/lib/openh264/WelsVP.lib
        ${SOURCE_DIR}/lib/openh264/encConsole.lib
      )
    endif()
  endif()
endif()