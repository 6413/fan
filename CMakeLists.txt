cmake_minimum_required(VERSION 3.20)
project(fan)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

set(LOCO_BOX2D ON)
set(LOCO_IMGUI ON)
set(LOCO_JSON ON)
set(LOCO_ASSIMP ON)
set(LOCO_OPENGL ON)
set(LOCO_VULKAN ON)
set(BUILD_MAIN ON)

set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MD -MT -L/usr/local/lib/ -Wall -Wextra -fpermissive -ferror-limit=4 -stdlib=libstdc++ -fsized-deallocation -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shift-op-parentheses -Wno-unused-variable -Wno-int-to-void-pointer-cast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-function -Wno-bitwise-op-parentheses")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-offsetof -Wno-missing-field-initializers -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-parameter -Wno-unused-value")
set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=gold")

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/fan/)
set(OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/fan)
file(MAKE_DIRECTORY ${OUTPUT_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

set(BUILD_SHARED_LIBS ON)

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

add_library(fmt STATIC ${ROOT_PATH}/fmt.cpp)
set_target_properties(fmt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "fmt"
)

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

file(GLOB_RECURSE FAN_SOURCES 
    ${ROOT_PATH}/*.cpp
    ${ROOT_PATH}/*/*.cpp
    ${ROOT_PATH}/*/*/*.cpp
    ${ROOT_PATH}/*/*/*/*.cpp
)

list(FILTER FAN_SOURCES EXCLUDE REGEX ".*imgui/.*")
list(FILTER FAN_SOURCES EXCLUDE REGEX ".*nativefiledialog/.*")
if(NOT LOCO_BOX2D)
    list(FILTER FAN_SOURCES EXCLUDE REGEX ".*physics_shapes\\.cpp$")
    list(FILTER FAN_SOURCES EXCLUDE REGEX ".*b2_integration\\.cpp$")
endif()

# compile all fan sources into an object library
add_library(fan_obj OBJECT ${FAN_SOURCES})
apply_loco_definitions(fan_obj)

# link the object library into the shared library
add_library(fan SHARED $<TARGET_OBJECTS:fan_obj>)
apply_loco_definitions(fan)
set_target_properties(fan PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_PATH}
    OUTPUT_NAME "fan"
)

if(BUILD_MAIN)
    if(NOT DEFINED MAIN)
        set(MAIN examples/graphics/2D/shapes/rectangle.cpp)
    endif()
    
    add_executable(a.exe ${MAIN} $<TARGET_OBJECTS:fan_obj>)
    apply_loco_definitions(a.exe)
    
    if(UNIX)
        target_link_libraries(a.exe PUBLIC
            fan
            fmt
            nfd
            webp
            glfw
            X11
            opus
            pulse-simple
            uv
            GL
            GLEW
            ${GTK3_LIBRARIES}
        )
				if (LOCO_BOX2D MATCHES ON)
					target_link_libraries(a.exe PUBLIC box2d)
				endif()

        if(LOCO_IMGUI MATCHES ON)
            target_link_libraries(a.exe PUBLIC imgui)
        endif()
        if(LOCO_ASSIMP MATCHES ON)
            target_link_libraries(a.exe PUBLIC assimp)
        endif()
				if(LOCO_VULKAN MATCHES ON)
            target_link_libraries(a.exe PUBLIC vulkan)
						find_package(Vulkan REQUIRED)
						find_library(SHADERC_LIB shaderc_shared PATHS /usr/local/lib)
						target_link_libraries(a.exe PUBLIC Vulkan::Vulkan ${SHADERC_LIB})
        endif()
    else()
        message(WARNING "Windows backend not implemented")
    endif()
endif()
