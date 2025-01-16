cmake_minimum_required(VERSION 3.15)
project(fan)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Response file settings for large command lines
set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

# Define build options
option(LOCO_BOX2D "Enable Box2D" ON)
option(LOCO_IMGUI "Enable ImGui" ON)
option(LOCO_JSON "Enable JSON" ON)
option(LOCO_ASSIMP "Enable Assimp" ON)
option(BUILD_MAIN "Build main executable" ON)

# Set compiler and flags
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fPIC -fpermissive -ferror-limit=4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shift-op-parentheses -Wno-unused-variable -Wno-int-to-void-pointer-cast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-function -Wno-bitwise-op-parentheses")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof -Wno-missing-field-initializers -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-parameter -Wno-unused-value")
set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=gold")

# Define paths and create output directory
set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/fan)
file(MAKE_DIRECTORY ${OUTPUT_PATH})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Box2D
include(FetchContent)
FetchContent_Declare(
    box2d
    GIT_REPOSITORY https://github.com/erincatto/box2d.git
    GIT_TAG v3.0.0
)
set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(box2d)

# Common compile definitions function
function(apply_loco_definitions target)
    if(LOCO_IMGUI)
        target_compile_definitions(${target} PRIVATE -Dloco_imgui)
    endif()
    if(LOCO_JSON)
        target_compile_definitions(${target} PRIVATE -Dloco_json)
    endif()
    if(LOCO_ASSIMP)
        target_compile_definitions(${target} PRIVATE -Dloco_assimp)
    endif()
    if(LOCO_BOX2D)
        target_compile_definitions(${target} PRIVATE -Dloco_box2d)
    endif()
    target_compile_definitions(${target} PRIVATE
        IMGUI_IMPL_OPENGL_LOADER_CUSTOM
        IMGUI_DEFINE_MATH_OPERATORS
    )
endfunction()

# FMT library
add_library(fmt STATIC ${ROOT_PATH}/fmt.cpp)
set_target_properties(fmt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "fmt"
)

# ImGui library
if(LOCO_IMGUI)
    file(GLOB IMGUI_SOURCES ${ROOT_PATH}/imgui/*.cpp)
    add_library(imgui STATIC ${IMGUI_SOURCES})
    target_include_directories(imgui PRIVATE ${ROOT_PATH}/imgui)
    target_compile_definitions(imgui PRIVATE 
        IMGUI_IMPL_OPENGL_LOADER_CUSTOM
        IMGUI_DEFINE_MATH_OPERATORS
        IMGUI_DISABLE_SSE
        STBI_NO_SIMD
    )
    apply_loco_definitions(imgui)
    set_target_properties(imgui PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
        OUTPUT_NAME "imgui"
    )
endif()

# Native File Dialog library
if(UNIX)
    file(GLOB NFD_SOURCES ${ROOT_PATH}/nativefiledialog/*.c)
    list(FILTER NFD_SOURCES EXCLUDE REGEX ".*nfd_zenity\\.c$")
    list(FILTER NFD_SOURCES EXCLUDE REGEX ".*nfd_win\\.cpp$")
else()
    file(GLOB NFD_SOURCES ${ROOT_PATH}/nativefiledialog/*.c ${ROOT_PATH}/nativefiledialog/*.cpp)
    list(FILTER NFD_SOURCES EXCLUDE REGEX ".*nfd_zenity\\.c$")
    list(FILTER NFD_SOURCES EXCLUDE REGEX ".*nfd_gtk\\.c$")
endif()

add_library(nfd STATIC ${NFD_SOURCES})
if(UNIX)
    target_include_directories(nfd PRIVATE ${GTK3_INCLUDE_DIRS})
endif()
set_target_properties(nfd PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "nfd"
)

# Main fan library
file(GLOB_RECURSE FAN_SOURCES 
    ${ROOT_PATH}/*.cpp
    ${ROOT_PATH}/*/*.cpp
    ${ROOT_PATH}/*/*/*.cpp
    ${ROOT_PATH}/*/*/*/*.cpp
)
list(FILTER FAN_SOURCES EXCLUDE REGEX ".*imgui/.*")
list(FILTER FAN_SOURCES EXCLUDE REGEX ".*nativefiledialog/.*")
list(FILTER FAN_SOURCES EXCLUDE REGEX ".*fmt\\.cpp$")

add_library(fan SHARED ${FAN_SOURCES})
apply_loco_definitions(fan)
set_target_properties(fan PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "fan"
)

# Main executable
if(BUILD_MAIN)
    if(NOT DEFINED MAIN)
        set(MAIN examples/graphics/2D/shapes/rectangle.cpp)
    endif()
    
    add_executable(a.exe ${MAIN})
    
    if(UNIX)
        target_link_libraries(a.exe
            fan
            imgui
            nfd
            fmt
            box2d
            webp
            glfw
            X11
            assimp
            opus
            pulse-simple
            uv
            ${GTK3_LIBRARIES}
        )
    else()
        message(WARNING "Windows backend not implemented")
    endif()
endif()