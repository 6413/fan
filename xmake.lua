set_project("fan")
set_languages("cxx23")

set_toolchains("clang")

rule("mode.mode_none") rule_end()

add_rules("mode.mode_none", "mode.debug", "mode.release")
set_defaultmode("mode_none")

if is_mode("mode_none") then
    add_defines("_DEBUG=3")
end

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

option("fan_gui")           set_default(true) option_end()
option("fan_physics")       set_default(true) option_end()
option("fan_json")          set_default(true) option_end()
option("fan_3d")            set_default(false) option_end()
option("fan_opengl")        set_default(true)  option_end()
option("fan_vulkan")        set_default(false) option_end()
option("fan_fmt")           set_default(true) option_end()
option("fan_wayland_screen")set_default(false) option_end()
option("fan_network")       set_default(false) option_end()

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

add_includedirs(".", {public = true})

local llvm_sdk = "/usr/lib/llvm"
if type(get_config) == "function" then
    local cfgsdk = get_config("sdk")
    if cfgsdk and cfgsdk ~= "" then llvm_sdk = cfgsdk end
end

add_cxxflags("-stdlib=libstdc++", {force = true})

set_values("c++.modules.std", false)
set_values("c++.modules.compat", false)

add_cxxflags("-pthread", {force = true})

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

table.insert(module_files, "fan/fan.ixx")

function find_impl_files(module_list)
    local impl_files = {}
    
    for _, module_path in ipairs(module_list) do
        local dir = path.directory(module_path)
        local name = path.basename(module_path)
        
        local impl_path = path.join(dir, name .. "_impl.cpp")
        if os.isfile(impl_path) then
            table.insert(impl_files, impl_path)
        end
    end
    
    return impl_files
end

local impl_files = find_impl_files(module_files)

if os.isfile("fan/graphics/algorithm/AStar.cpp") then
    table.insert(impl_files, "fan/graphics/algorithm/AStar.cpp")
end

target("fan_modules")
    set_kind("static")
    
    add_files(module_files)
    
    for _, impl in ipairs(impl_files) do
        add_files(impl)
    end
    
    add_includedirs(".", "thirdparty/fan/include", {public = true})
target_end()

if has_config("fan_gui") then
    target("imgui")
        set_kind("static")
        add_cxxflags("-stdlib=libstdc++", {force = true})
				add_ldflags("-stdlib=libstdc++", "-lstdc++", {force = true})
        add_includedirs(
            "fan/imgui",
            "fan/imgui/misc/freetype",
            "thirdparty/fan/include",
            "thirdparty/fan/include/freetype2"
        )
        
        on_load(function (target)
            if target:is_plat("linux") then
                import("lib.detect.find_tool")
                local pkg_config = find_tool("pkg-config")
                if pkg_config then
                    local result = os.iorunv("pkg-config", {"--cflags-only-I", "glib-2.0"})
                    if result then
                        for _, path in ipairs(result:split("%s+")) do
                            if path:startswith("-I") then
                                target:add("includedirs", path:sub(3))
                            end
                        end
                    end
                end
            end
        end)
        
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
    
    target("nfd")
        set_kind("static")
        
        if is_plat("linux") then
            add_files(
                "fan/nativefiledialog/nfd_common.c",
                "fan/nativefiledialog/nfd_gtk.c"
            )
        elseif is_plat("windows") then
            add_files(
                "fan/nativefiledialog/nfd_common.c",
                "fan/nativefiledialog/nfd_win.cpp"
            )
        end
        
        on_load(function (target)
            if target:is_plat("linux") then
                import("lib.detect.find_tool")
                local pkg_config = find_tool("pkg-config")
                if pkg_config then
                    local cflags = os.iorunv("pkg-config", {"--cflags-only-I", "gtk+-3.0"})
                    if cflags then
                        for _, path in ipairs(cflags:split("%s+")) do
                            if path:startswith("-I") then
                                target:add("includedirs", path:sub(3))
                            end
                        end
                    end
                    
                    local libs = os.iorunv("pkg-config", {"--libs", "gtk+-3.0"})
                    if libs then
                        for _, lib in ipairs(libs:split("%s+")) do
                            if lib:startswith("-l") then
                                target:add("links", lib:sub(3))
                            end
                        end
                    end
                end
            end
        end)
    target_end()
end

target("a.exe")
    set_kind("binary")
    
    add_deps("fan_modules")
    
    if has_config("fan_gui") then
        add_deps("imgui", "nfd")
    end
    
    add_files(module_files)
    
    add_files("examples/engine_demos/engine_demo.cpp", {module = false})
    
    add_includedirs(".", {public = true})
    
    add_linkdirs("thirdparty/fan/lib")
    
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
        end
        
        if has_config("fan_physics") then
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
    
    on_load(function (target)
        if target:is_plat("linux") and has_config("fan_gui") then
            import("lib.detect.find_tool")
            local pkg_config = find_tool("pkg-config")
            if pkg_config then
                local libs = os.iorunv("pkg-config", {"--libs", "gtk+-3.0"})
                if libs then
                    for _, lib in ipairs(libs:split("%s+")) do
                        if lib:startswith("-l") then
                            target:add("links", lib:sub(3))
                        end
                    end
                end
            end
        end
    end)
target_end()

local marker = "fan_modules_info_printed.flag"

after_load(function (target)
    if target:name() ~= "fan_modules" then
        return
    end

    if os.isfile(marker) then
        return
    end

    io.writefile(marker, "1")

    print("Module files: " .. #module_files)
    print("Implementation files: " .. #impl_files)

    if #impl_files > 0 then
        print("Found implementations:")
        for i = 1, math.min(5, #impl_files) do
            print("  - " .. impl_files[i])
        end
        if #impl_files > 5 then
            print("  ... and " .. (#impl_files - 5) .. " more")
        end
    end
end)
