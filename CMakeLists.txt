cmake_minimum_required(VERSION 3.10)

project(fan_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Add the include directories
include_directories(./)

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
if(NOT DEFINED MAIN)
	set(MAIN examples/graphics/2D/shapes/rectangle.cpp)
endif()
set(OUTPUT a.exe)

# Function to check if a directory exists
function(check_directory_exists DIR)
    if (EXISTS "${DIR}")
        set(EXISTS TRUE PARENT_SCOPE)
    else()
        set(EXISTS FALSE PARENT_SCOPE)
    endif()
endfunction()

# Check for the fan directory in various locations
if(NOT DEFINED ROOT_PATH)
    set(SEARCH_PATHS "/usr/include/fan/" "/usr/local/include/fan/" "${CMAKE_SOURCE_DIR}/fan/")
    
    set(DIR_FOUND FALSE)
    foreach(PATH ${SEARCH_PATHS})
        check_directory_exists(${PATH})
        if(EXISTS)
            set(ROOT_PATH ${PATH})
            set(DIR_FOUND TRUE)
            break()
        endif()
    endforeach()
    
    if(NOT DIR_FOUND)
        message(FATAL_ERROR "fan directory not found in any of the specified locations.")
    endif()
endif()

set(ENABLE_IMGUI ON)

set(CMAKE_CXX_FLAGS "-Winvalid-pch -w ${CMAKE_CXX_FLAGS} -I ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_PATH}../ -ferror-limit=4 -Wno-int-to-void-pointer-cast -Wno-unused-parameter -Wno-unused-function -Wno-shift-op-parentheses -Wno-unused-variable")
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

if(DEFINED BUILD_FAN)
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

	list(FILTER SOURCES EXCLUDE REGEX "${ROOT_PATH}/fmt.cpp")
	
	message("Sources are: ${SOURCES}")
	
	add_library(fan OBJECT ${SOURCES})
	add_executable(${OUTPUT} ${MAIN} $<TARGET_OBJECTS:fan>)
endif()

message("Main is: ${MAIN}")

if(NOT DEFINED BUILD_FAN)
	add_executable(${OUTPUT} ${MAIN})
endif()

if(WIN32)
	message("Windows back-end undone.")
else()
	file(GLOB FAN_LIBS "/usr/local/lib/fan/*.a")
	message("fan libs: ${FAN_LIBS}")
	if(DEFINED BUILD_FAN)
		list(FILTER FAN_LIBS EXCLUDE REGEX ".*fan.a$")
	endif()
    # On Linux, use the system libraries
    target_link_libraries(${OUTPUT} webp glfw X11 ${FAN_LIBS})
endif()

link_directories(${CMAKE_SOURCE_DIR}/lib)