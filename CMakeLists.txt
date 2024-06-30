# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MyProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Add the include directories
include_directories(../ include/fan/ ./include/)

find_package(PkgConfig REQUIRED)

# Check for GTK
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Add GTK flags to the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GTK3_CFLAGS_OTHER}")

# Add the GTK include directories
include_directories(${GTK3_INCLUDE_DIRS})

# Link the GTK libraries
link_libraries(${GTK3_LIBRARIES})

set(CMAKE_CXX_COMPILER "clang++")

# Set the source files and output
set(MAIN examples/graphics/gui/model_maker.cpp)
set(OUTPUT a.exe)

# Set the paths
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(ROOT_PATH "./include/fan/")
endif()

set(PCH_SOURCE "${ROOT_PATH}/pch.cpp")

set(LINK_PATH "lib/fan/")

# Set the precompiled header file
set(PCH_NAME "pch.h")

set(ENABLE_IMGUI ON)

# Set the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CURRENT_SOURCE_DIR}/${LINK_PATH}/pch.o -include-pch ${CMAKE_CURRENT_SOURCE_DIR}/${LINK_PATH}/pch.h.gch -Winvalid-pch -w -I /mnt/c/Users/0b347/Documents/GitHub/ -I ${ROOT_PATH}/../ ${CMAKE_CXX_FLAGS} -ferror-limit=4 -Wno-int-to-void-pointer-cast -Wno-unused-parameter -Wno-unused-function -Wno-shift-op-parentheses -Wno-unused-variable")
set(DEBUG_FLAGS "-g")
set(RELEASE_FLAGS "-s -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3 -fdata-sections -ffunction-sections -march=native -Wl,--gc-sections")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")

if(ENABLE_IMGUI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dloco_imgui -DIMGUI_IMPL_OPENGL_LOADER_CUSTOM -DIMGUI_DEFINE_MATH_OPERATORS")
endif()

# loco_json
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dloco_json")

message("Root path is: ${ROOT_PATH}")

file(GLOB_RECURSE SOURCES "${ROOT_PATH}/*.cpp" "${ROOT_PATH}/*.c")

# Separate the ImGui sources
file(GLOB_RECURSE IMGUI_SOURCES "${ROOT_PATH}/imgui/*.cpp" "${ROOT_PATH}/imgui/*.c")

# Separate the nativefiledialog sources
file(GLOB_RECURSE NFD_SOURCES "${ROOT_PATH}/nativefiledialog/*.cpp" "${ROOT_PATH}/nativefiledialog/*.c")

# Separate the fmt sources
set(FMT_SOURCES "${ROOT_PATH}/fmt.h" "${ROOT_PATH}/fmt.cpp")

# Remove the ImGui, nativefiledialog and fmt sources from the SOURCES variable
list(REMOVE_ITEM SOURCES ${IMGUI_SOURCES} ${NFD_SOURCES} ${FMT_SOURCES})

list(FILTER NFD_SOURCES EXCLUDE REGEX "nfd_zenity.c")
list(REMOVE_ITEM SOURCES "nfd_zenity.c")

list(FILTER SOURCES EXCLUDE REGEX ${PCH_SOURCE})
list(REMOVE_ITEM SOURCES ${PCH_SOURCE})

list(FILTER SOURCES EXCLUDE REGEX "${ROOT_PATH}/fmt.cpp")
message("Sources is: ${SOURCES}")

# Compile all source files into object files
add_library(fan OBJECT ${SOURCES})

# Add the executable
add_executable(${OUTPUT} ${MAIN} $<TARGET_OBJECTS:fan>)

#target_precompile_headers(fan PRIVATE ${ROOT_PATH}/pch.h ${ROOT_PATH}/graphics/loco.h)


# Conditionally set the libraries based on the platform
if(WIN32)
    # On Windows, use the precompiled .lib files
    link_directories(${CMAKE_SOURCE_DIR}/lib)
    target_link_libraries(${OUTPUT} fan)
else()
    # On Linux, use the system libraries
    target_link_libraries(${OUTPUT} webp glfw X11 ${CMAKE_CURRENT_SOURCE_DIR}/lib/fan/libimgui.a ${CMAKE_CURRENT_SOURCE_DIR}/lib/fan/libnfd.a ${CMAKE_CURRENT_SOURCE_DIR}/lib/fan/libfmt.a)
endif()

# Tell CMake where to find the libraries
link_directories(${CMAKE_SOURCE_DIR}/lib)

# Link the libraries
target_link_libraries(${OUTPUT} fan)
