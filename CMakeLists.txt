cmake_minimum_required(VERSION 3.10)

project(fan_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)

set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

# Add the include directories
include_directories(./)

find_package(PkgConfig REQUIRED)

# Check for GTK
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Add GTK flags to the compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GTK3_CFLAGS_OTHER}")
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=gold")

if (CMAKE_GENERATOR MATCHES "Ninja")
    file(
        WRITE "${CMAKE_BINARY_DIR}/GNUMakeRulesOverwrite.cmake" 
            "STRING(REPLACE \"-MD\" \"-MMD\" CMAKE_DEPFILE_FLAGS_C \"\${CMAKE_DEPFILE_FLAGS_C}\")\n"
            "STRING(REPLACE \"-MD\" \"-MMD\" CMAKE_DEPFILE_FLAGS_CXX \"\${CMAKE_DEPFILE_FLAGS_CXX}\")\n"
    )
    set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_BINARY_DIR}/GNUMakeRulesOverwrite.cmake" CACHE INTERNAL "")
endif()

# Add the GTK include directories
include_directories(${GTK3_INCLUDE_DIRS})

# Link the GTK libraries
link_libraries(${GTK3_LIBRARIES})
link_directories(${CMAKE_SOURCE_DIR}/lib)
link_directories(/usr/local/lib/fan)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_PATH}../ -ferror-limit=4 -Wno-int-to-void-pointer-cast -Wno-unused-parameter -Wno-unused-function -Wno-shift-op-parentheses -Wno-unused-variable -Wno-int-to-void-pointer-cast -Wno-unused-parameter -Wno-unused-function -Wno-shift-op-parentheses -Wno-unused-variable -Wno-bitwise-op-parentheses -Wno-invalid-offsetof -Wno-missing-field-initializers -Wno-sign-compare -Wno-unused-but-set-parameter -Wno-unused-value")
set(DEBUG_FLAGS "-gdwarf-4")
set(RELEASE_FLAGS "-s -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -O3 -fdata-sections -ffunction-sections -march=native -Wl,--gc-sections")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

if(ENABLE_IMGUI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dloco_imgui -DIMGUI_IMPL_OPENGL_LOADER_CUSTOM -DIMGUI_DEFINE_MATH_OPERATORS -DIMGUI_DISABLE_DEBUG_TOOLS")
endif()

# loco_json
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dloco_json -Dloco_assimp")

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
	
		message("Libs are: ${FAN_LIBS}")
    # On Linux, use the system libraries
    target_link_libraries(${OUTPUT} webp glfw X11 assimp opus pulse-simple uv fan.so ${FAN_LIBS})
endif()
