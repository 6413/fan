set_project("fan")
set_languages("cxx23")

set_toolchains("clang-20")

rule("mode.mode_none") rule_end()

add_rules("mode.mode_none", "mode.debug", "mode.release")
set_defaultmode("mode_none")

if is_mode("release") then
    set_optimize("fastest")
    add_cxxflags("-O3", {force = true})
    add_ldflags("-s", {force = true})
    add_defines("NDEBUG")
elseif is_mode("debug") then
    set_optimize("none")
    set_symbols("debug")
    add_cxxflags("-g", "-gdwarf-4", {force = true})
    add_defines("_DEBUG=4")
end

-- Options
option("fan_gui")           set_default(true) option_end()
option("fan_physics")       set_default(true) option_end()
option("fan_json")          set_default(true) option_end()
option("fan_3d")            set_default(false) option_end()
option("fan_opengl")        set_default(true)  option_end()
option("fan_vulkan")        set_default(false) option_end()
option("fan_fmt")           set_default(true) option_end()
option("fan_wayland_screen")set_default(false) option_end()
option("fan_network")       set_default(false) option_end()

-- Defines based on options
add_defines("fan_opengl")
if has_config("fan_gui") then
    add_defines("fan_gui")
end
if has_config("fan_json") then
    add_defines("fan_json")
end
if has_config("fan_3d") then
    add_defines("fan_3D")
end
if has_config("fan_physics") then
    add_defines("fan_physics")
end
if has_config("fan_vulkan") then
    add_defines("fan_vulkan")
end
if has_config("fan_network") then
    add_defines("fan_network")
end
if has_config("fan_fmt") then
    add_defines("fan_fmt")
end

-- LLVM SDK setup
local llvm_sdk = "/usr/lib/llvm-20"
if type(get_config) == "function" then
    local cfgsdk = get_config("sdk")
    if cfgsdk and cfgsdk ~= "" then llvm_sdk = cfgsdk end
end

if os.isdir(llvm_sdk) then
    add_cxxflags("-stdlib=libc++", "-I" .. llvm_sdk .. "/include/c++/v1", {force = true})
    add_cxxflags("-resource-dir=" .. llvm_sdk .. "/lib/clang/20", {force = true})
    add_ldflags("-L" .. llvm_sdk .. "/lib", "-lc++", "-lc++abi", {force = true})
else
    add_cxxflags("-stdlib=libc++", {force = true})
    add_ldflags("-stdlib=libc++ -lc++abi", {force = true})
end

set_values("c++.modules.std", false)
set_values("c++.modules.compat", false)

add_cxxflags("-pthread", {force = true})

-- Warning suppression
add_cxxflags(
    "-Wall", "-Wextra", "-ferror-limit=20",
    "-Wno-shift-op-parentheses",
    "-Wno-unused-variable",
    "-Wno-int-to-void-pointer-cast",
    "-Wno-unused-parameter",
    "-Wno-unused-function",
    "-Wno-bitwise-op-parentheses",
    "-Wno-invalid-offsetof",
    "-Wno-missing-field-initializers",
    "-Wno-sign-compare",
    "-Wno-unused-but-set-parameter",
    "-Wno-unused-value",
    "-w",
    "-fsized-deallocation",
    "-fmacro-backtrace-limit=0"
)

if has_config("fan_gui") then
    add_defines(
        "IMGUI_IMPL_OPENGL_LOADER_CUSTOM",
        "IMGUI_DEFINE_MATH_OPERATORS",
        "IMGUI_DISABLE_SSE",
        "IMGUI_ENABLE_FREETYPE",
        "IMGUI_ENABLE_FREETYPE_LUNASVG",
        "STBI_NO_SIMD"
    )
end

-- All module files (XMake automatically determines build order from dependencies)
local module_files = {
    "fan/types/types.ixx",
    "fan/types/magic.ixx",
    "fan/types/color.ixx",
    "fan/types/vector.ixx",
    "fan/types/quaternion.ixx",
    "fan/types/matrix.ixx",
    "fan/types/masterpiece.ixx",
    "fan/types/fstring.ixx",
    "fan/math/math.ixx",
    "fan/time.ixx",
    "fan/utility.ixx",
    "fan/print.ixx",
    "fan/random.ixx",
    "fan/io/directory.ixx",
    "fan/io/file.ixx",
    "fan/physics/collision/rectangle.ixx",
    "fan/physics/collision/triangle.ixx",
    "fan/physics/collision/circle.ixx",
    "fan/physics/physics_types.ixx",
    "fan/graphics/common_types.ixx",
    "fan/graphics/camera.ixx",
    "fan/graphics/image_load.ixx",
    "fan/graphics/webp.ixx",
    "fan/graphics/stb.ixx",
    "fan/graphics/common_context.ixx",
    "fan/graphics/opengl/gl_core.ixx",
    "fan/graphics/opengl/uniform_block.ixx",
    "fan/texture_pack/tp0.ixx",
    "fan/graphics/shapes.ixx",
    "fan/graphics/loco.ixx",
    "fan/graphics/graphics.ixx",
    "fan/graphics/file_dialog.ixx",
    "fan/graphics/algorithm/raycast_grid.ixx",
    "fan/graphics/algorithm/pathfind.ixx",
    "fan/window/window.ixx",
    "fan/window/input_common.ixx",
    "fan/window/input.ixx",
    "fan/window/input_action.ixx",
    "fan/audio/audio.ixx",
    "fan/event/event_types.ixx",
    "fan/event/event.ixx",
    "fan/network/network.ixx",
    "fan/graphics/graphics_network.ixx",
    "fan/noise.ixx"
}

-- Add conditional modules
if has_config("fan_wayland_screen") then
    table.insert(module_files, "fan/video/screen_codec.ixx")
end

if has_config("fan_json") then
    table.insert(module_files, "fan/types/json.ixx")
end

if has_config("fan_fmt") or has_config("fan_gui") then
    table.insert(module_files, "fan/fmt.ixx")
end

if has_config("fan_gui") then
    table.insert(module_files, "fan/graphics/gui/gui_base.ixx")
    table.insert(module_files, "fan/graphics/gui/text_logger.ixx")
    table.insert(module_files, "fan/graphics/gui/gui_types.ixx")
    table.insert(module_files, "fan/graphics/gui/gui.ixx")
    table.insert(module_files, "fan/graphics/gui/console.ixx")
    table.insert(module_files, "fan/graphics/gui/tilemap_editor/loader.ixx")
    table.insert(module_files, "fan/graphics/gui/tilemap_editor/renderer0.ixx")
end

if has_config("fan_physics") then
    table.insert(module_files, "fan/physics/b2_integration.ixx")
    table.insert(module_files, "fan/physics/physics_common_context.ixx")
    table.insert(module_files, "fan/graphics/physics_shapes.ixx")
end

if has_config("fan_3d") then
    table.insert(module_files, "fan/graphics/opengl/3D/objects/fms.ixx")
    table.insert(module_files, "fan/graphics/opengl/3D/objects/model.ixx")
end

-- Add main fan module at the end
table.insert(module_files, "fan/fan.ixx")

-- Function to find corresponding implementation files
function find_impl_files(module_list)
    local impl_files = {}
    
    for _, module_path in ipairs(module_list) do
        -- Get directory and filename without extension
        local dir = path.directory(module_path)
        local name = path.basename(module_path)
        
        -- Check for _impl.cpp variant
        local impl_path = path.join(dir, name .. "_impl.cpp")
        if os.isfile(impl_path) then
            table.insert(impl_files, impl_path)
        end
    end
    
    return impl_files
end

-- Find all implementation files dynamically
local impl_files = find_impl_files(module_files)

-- Always add AStar.cpp if it exists
if os.isfile("fan/graphics/algorithm/AStar.cpp") then
    table.insert(impl_files, "fan/graphics/algorithm/AStar.cpp")
end

-- Target: fan_modules
target("fan_modules")
    set_kind("static")
    
    -- Add all module files
    add_files(module_files)
    
    -- Add dynamically found implementation files
    for _, impl in ipairs(impl_files) do
        add_files(impl)
    end
    
    add_includedirs(".", "thirdparty/fan/include", {public = true})
target_end()

-- Target: imgui (if GUI enabled)
if has_config("fan_gui") then
    target("imgui")
        set_kind("static")
        
        add_includedirs(
            "fan/imgui",
            "fan/imgui/misc/freetype",
            "thirdparty/fan/include",
            "thirdparty/fan/include/freetype2"
        )
        
        add_files(
            "fan/imgui/imgui.cpp",
            "fan/imgui/imgui_draw.cpp",
            "fan/imgui/imgui_widgets.cpp",
            "fan/imgui/imgui_tables.cpp",
            "fan/imgui/imgui_impl_glfw.cpp",
            "fan/imgui/imgui_impl_opengl3.cpp",
            "fan/imgui/implot_items.cpp",
            "fan/imgui/implot.cpp",
            "fan/imgui/text_editor.cpp",
            "fan/imgui/misc/freetype/imgui_freetype.cpp"
        )
        
        if has_config("fan_vulkan") then
            add_files("fan/imgui/imgui_impl_vulkan.cpp")
        end
        
        if has_config("fan_3d") and has_config("fan_gui") then
            add_files("fan/imgui/ImGuizmo.cpp")
        end
        
        add_linkdirs("thirdparty/fan/lib")
        add_links("freetype", "lunasvg")
    target_end()
    
    -- Target: nfd
    target("nfd")
        set_kind("static")
        
        if is_plat("linux") then
            add_files(
                "fan/nativefiledialog/nfd_common.c",
                "fan/nativefiledialog/nfd_gtk.c"
            )
            
            -- Add GTK3 include paths manually (complete set)
            add_includedirs(
                "/usr/include/gtk-3.0",
                "/usr/include/at-spi2-atk/2.0",
                "/usr/include/at-spi-2.0",
                "/usr/include/dbus-1.0",
                "/usr/lib/x86_64-linux-gnu/dbus-1.0/include",
                "/usr/include/gio-unix-2.0",
                "/usr/include/cairo",
                "/usr/include/pango-1.0",
                "/usr/include/harfbuzz",
                "/usr/include/fribidi",
                "/usr/include/atk-1.0",
                "/usr/include/pixman-1",
                "/usr/include/uuid",
                "/usr/include/freetype2",
                "/usr/include/gdk-pixbuf-2.0",
                "/usr/include/libpng16",
                "/usr/include/libmount",
                "/usr/include/blkid",
                "/usr/include/glib-2.0",
                "/usr/lib/x86_64-linux-gnu/glib-2.0/include"
            )
            
            add_links("gtk-3", "gdk-3", "pangocairo-1.0", "pango-1.0", 
                      "harfbuzz", "atk-1.0", "cairo-gobject", "cairo",
                      "gdk_pixbuf-2.0", "gio-2.0", "gobject-2.0", "glib-2.0")
            
        elseif is_plat("windows") then
            add_files(
                "fan/nativefiledialog/nfd_common.c",
                "fan/nativefiledialog/nfd_win.cpp"
            )
        end
    target_end()
end

-- Target: main executable
target("a.exe")
    set_kind("binary")
    
    add_deps("fan_modules")
    
    if has_config("fan_gui") then
        add_deps("imgui", "nfd")
    end
    
    -- Add all module files to the executable (required for module linking)
    add_files(module_files)
    
    -- Add main source file
    add_files("examples/engine_demos/engine_demo.cpp", {module = false})
    
    add_includedirs(".", {public = true})
    
    -- Link directories
    add_linkdirs("thirdparty/fan/lib", "lib/fan")
    
    -- Common libraries
    if is_plat("linux") then
        add_links(
            "webp", "glfw", "X11", "opus", "pulse-simple",
            "uv", "GL", "GLEW", "ssl", "crypto", "png", "z", "curl"
        )
        
        if has_config("fan_fmt") then
            add_links("fmt")
        end
        
        if has_config("fan_gui") then
            add_links("freetype", "lunasvg")
            
            -- Add GTK3 libraries for the executable too
            add_links("gtk-3", "gdk-3", "pangocairo-1.0", "pango-1.0", 
                      "harfbuzz", "atk-1.0", "cairo-gobject", "cairo",
                      "gdk_pixbuf-2.0", "gio-2.0", "gobject-2.0", "glib-2.0")
        end
        
        if has_config("fan_physics") then
            -- Box2D must be linked with whole-archive flags
            add_ldflags("-Wl,--whole-archive", "thirdparty/fan/lib/libbox2d.a", "-Wl,--no-whole-archive", {force = true})
        end
        
        if has_config("fan_3d") then
            add_links("assimp")
        end
        
        if has_config("fan_vulkan") then
            add_packages("vulkansdk")
            add_links("shaderc_shared")
        end
        
        if has_config("fan_wayland_screen") then
            add_links("wayland-client", "pipewire-0.3", "dbus-1")
            add_links("avcodec", "avutil", "swscale")
        end
        
        add_ldflags("-fuse-ld=gold", {force = true})
        
    elseif is_plat("windows") then
        add_links("opengl32")
        add_linkdirs("lib/GLFW", "lib/GLEW", "lib/libuv", "lib/libwebp", "lib/opus", "lib/openssl")
        
        add_links(
            "glfw3_mt", "glew32s", "uv_a", "libwebp",
            "opus", "libssl", "libcrypto"
        )
        
        if has_config("fan_gui") then
            add_linkdirs("lib/freetype", "lib/lunasvg")
            add_links("freetype", "lunasvg")
        end
        
        if has_config("fan_physics") then
            add_linkdirs("lib/box2d")
            add_links("box2d")
        end
        
        if has_config("fan_3d") then
            add_linkdirs("C:/Program Files/Assimp/lib/x64")
            add_links("assimp-vc143-mt")
        end
        
        if has_config("fan_vulkan") then
            add_packages("vulkansdk")
            add_links("shaderc_shared")
        end
        
        if has_config("fan_wayland_screen") then
            add_linkdirs("lib/libx264", "lib/openh264")
            add_links(
                "DXGI", "D3D11", "libx264",
                "welsdcore", "welsecore", "WelsDecPlus",
                "WelsEncPlus", "WelsVP"
            )
        end
    end
target_end()

-- Print module and implementation summary
after_load(function (target)
    if target:name() == "fan_modules" then
        print("Module files: " .. #module_files)
        print("Implementation files: " .. #impl_files)
        
        -- Print first 5 impl files as example
        if #impl_files > 0 then
            print("Found implementations:")
            for i = 1, math.min(5, #impl_files) do
                print("  - " .. impl_files[i])
            end
            if #impl_files > 5 then
                print("  ... and " .. (#impl_files - 5) .. " more")
            end
        end
    end
end)