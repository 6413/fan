#pragma once

#include <cstdint>

namespace fan {
  namespace opengl {
    static constexpr uint32_t GL_ACCUM = 0x0100;
    static constexpr uint32_t GL_ACCUM_ALPHA_BITS = 0x0D5B;
    static constexpr uint32_t GL_ACCUM_BLUE_BITS = 0x0D5A;
    static constexpr uint32_t GL_ACCUM_BUFFER_BIT = 0x00000200;
    static constexpr uint32_t GL_ACCUM_CLEAR_VALUE = 0x0B80;
    static constexpr uint32_t GL_ACCUM_GREEN_BITS = 0x0D59;
    static constexpr uint32_t GL_ACCUM_RED_BITS = 0x0D58;
    static constexpr uint32_t GL_ACTIVE_ATOMIC_COUNTER_BUFFERS = 0x92D9;
    static constexpr uint32_t GL_ACTIVE_ATTRIBUTES = 0x8B89;
    static constexpr uint32_t GL_ACTIVE_ATTRIBUTE_MAX_LENGTH = 0x8B8A;
    static constexpr uint32_t GL_ACTIVE_PROGRAM = 0x8259;
    static constexpr uint32_t GL_ACTIVE_RESOURCES = 0x92F5;
    static constexpr uint32_t GL_ACTIVE_SUBROUTINES = 0x8DE5;
    static constexpr uint32_t GL_ACTIVE_SUBROUTINE_MAX_LENGTH = 0x8E48;
    static constexpr uint32_t GL_ACTIVE_SUBROUTINE_UNIFORMS = 0x8DE6;
    static constexpr uint32_t GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS = 0x8E47;
    static constexpr uint32_t GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH = 0x8E49;
    static constexpr uint32_t GL_ACTIVE_TEXTURE = 0x84E0;
    static constexpr uint32_t GL_ACTIVE_UNIFORMS = 0x8B86;
    static constexpr uint32_t GL_ACTIVE_UNIFORM_BLOCKS = 0x8A36;
    static constexpr uint32_t GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH = 0x8A35;
    static constexpr uint32_t GL_ACTIVE_UNIFORM_MAX_LENGTH = 0x8B87;
    static constexpr uint32_t GL_ACTIVE_VARIABLES = 0x9305;
    static constexpr uint32_t GL_ADD = 0x0104;
    static constexpr uint32_t GL_ADD_SIGNED = 0x8574;
    static constexpr uint32_t GL_ALIASED_LINE_WIDTH_RANGE = 0x846E;
    static constexpr uint32_t GL_ALIASED_POINT_SIZE_RANGE = 0x846D;
    static constexpr uint32_t GL_ALL_ATTRIB_BITS = 0xFFFFFFFF;
    static constexpr uint32_t GL_ALL_BARRIER_BITS = 0xFFFFFFFF;
    static constexpr uint32_t GL_ALL_SHADER_BITS = 0xFFFFFFFF;
    static constexpr uint32_t GL_ALPHA = 0x1906;
    static constexpr uint32_t GL_ALPHA12 = 0x803D;
    static constexpr uint32_t GL_ALPHA16 = 0x803E;
    static constexpr uint32_t GL_ALPHA4 = 0x803B;
    static constexpr uint32_t GL_ALPHA8 = 0x803C;
    static constexpr uint32_t GL_ALPHA_BIAS = 0x0D1D;
    static constexpr uint32_t GL_ALPHA_BITS = 0x0D55;
    static constexpr uint32_t GL_ALPHA_INTEGER = 0x8D97;
    static constexpr uint32_t GL_ALPHA_SCALE = 0x0D1C;
    static constexpr uint32_t GL_ALPHA_TEST = 0x0BC0;
    static constexpr uint32_t GL_ALPHA_TEST_FUNC = 0x0BC1;
    static constexpr uint32_t GL_ALPHA_TEST_REF = 0x0BC2;
    static constexpr uint32_t GL_ALREADY_SIGNALED = 0x911A;
    static constexpr uint32_t GL_ALWAYS = 0x0207;
    static constexpr uint32_t GL_AMBIENT = 0x1200;
    static constexpr uint32_t GL_AMBIENT_AND_DIFFUSE = 0x1602;
    static constexpr uint32_t GL_AND = 0x1501;
    static constexpr uint32_t GL_AND_INVERTED = 0x1504;
    static constexpr uint32_t GL_AND_REVERSE = 0x1502;
    static constexpr uint32_t GL_ANY_SAMPLES_PASSED = 0x8C2F;
    static constexpr uint32_t GL_ANY_SAMPLES_PASSED_CONSERVATIVE = 0x8D6A;
    static constexpr uint32_t GL_ARRAY_BUFFER = 0x8892;
    static constexpr uint32_t GL_ARRAY_BUFFER_BINDING = 0x8894;
    static constexpr uint32_t GL_ARRAY_SIZE = 0x92FB;
    static constexpr uint32_t GL_ARRAY_STRIDE = 0x92FE;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BARRIER_BIT = 0x00001000;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER = 0x92C0;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS = 0x92C5;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES = 0x92C6;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_BINDING = 0x92C1;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE = 0x92C4;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_INDEX = 0x9301;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER = 0x90ED;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER = 0x92CB;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER = 0x92CA;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER = 0x92C8;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER = 0x92C9;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER = 0x92C7;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_SIZE = 0x92C3;
    static constexpr uint32_t GL_ATOMIC_COUNTER_BUFFER_START = 0x92C2;
    static constexpr uint32_t GL_ATTACHED_SHADERS = 0x8B85;
    static constexpr uint32_t GL_ATTRIB_STACK_DEPTH = 0x0BB0;
    static constexpr uint32_t GL_uint32_t_GENERATE_MIPMAP = 0x8295;
    static constexpr uint32_t GL_uint32_t_NORMAL = 0x0D80;
    static constexpr uint32_t GL_AUX0 = 0x0409;
    static constexpr uint32_t GL_AUX1 = 0x040A;
    static constexpr uint32_t GL_AUX2 = 0x040B;
    static constexpr uint32_t GL_AUX3 = 0x040C;
    static constexpr uint32_t GL_AUX_BUFFERS = 0x0C00;
    static constexpr uint32_t GL_BACK = 0x0405;
    static constexpr uint32_t GL_BACK_LEFT = 0x0402;
    static constexpr uint32_t GL_BACK_RIGHT = 0x0403;
    static constexpr uint32_t GL_BGR = 0x80E0;
    static constexpr uint32_t GL_BGRA = 0x80E1;
    static constexpr uint32_t GL_BGRA_INTEGER = 0x8D9B;
    static constexpr uint32_t GL_BGR_INTEGER = 0x8D9A;
    static constexpr uint32_t GL_BITMAP = 0x1A00;
    static constexpr uint32_t GL_BITMAP_TOKEN = 0x0704;
    static constexpr uint32_t GL_BLEND = 0x0BE2;
    static constexpr uint32_t GL_BLEND_COLOR = 0x8005;
    static constexpr uint32_t GL_BLEND_DST = 0x0BE0;
    static constexpr uint32_t GL_BLEND_DST_ALPHA = 0x80CA;
    static constexpr uint32_t GL_BLEND_DST_RGB = 0x80C8;
    static constexpr uint32_t GL_BLEND_EQUATION = 0x8009;
    static constexpr uint32_t GL_BLEND_EQUATION_ALPHA = 0x883D;
    static constexpr uint32_t GL_BLEND_EQUATION_RGB = 0x8009;
    static constexpr uint32_t GL_BLEND_SRC = 0x0BE1;
    static constexpr uint32_t GL_BLEND_SRC_ALPHA = 0x80CB;
    static constexpr uint32_t GL_BLEND_SRC_RGB = 0x80C9;
    static constexpr uint32_t GL_BLOCK_INDEX = 0x92FD;
    static constexpr uint32_t GL_BLUE = 0x1905;
    static constexpr uint32_t GL_BLUE_BIAS = 0x0D1B;
    static constexpr uint32_t GL_BLUE_BITS = 0x0D54;
    static constexpr uint32_t GL_BLUE_INTEGER = 0x8D96;
    static constexpr uint32_t GL_BLUE_SCALE = 0x0D1A;
    static constexpr uint32_t GL_BOOL = 0x8B56;
    static constexpr uint32_t GL_BOOL_VEC2 = 0x8B57;
    static constexpr uint32_t GL_BOOL_VEC3 = 0x8B58;
    static constexpr uint32_t GL_BOOL_VEC4 = 0x8B59;
    static constexpr uint32_t GL_BUFFER = 0x82E0;
    static constexpr uint32_t GL_BUFFER_ACCESS = 0x88BB;
    static constexpr uint32_t GL_BUFFER_ACCESS_FLAGS = 0x911F;
    static constexpr uint32_t GL_BUFFER_BINDING = 0x9302;
    static constexpr uint32_t GL_BUFFER_DATA_SIZE = 0x9303;
    static constexpr uint32_t GL_BUFFER_IMMUTABLE_STORAGE = 0x821F;
    static constexpr uint32_t GL_BUFFER_MAPPED = 0x88BC;
    static constexpr uint32_t GL_BUFFER_MAP_LENGTH = 0x9120;
    static constexpr uint32_t GL_BUFFER_MAP_OFFSET = 0x9121;
    static constexpr uint32_t GL_BUFFER_MAP_POINTER = 0x88BD;
    static constexpr uint32_t GL_BUFFER_SIZE = 0x8764;
    static constexpr uint32_t GL_BUFFER_STORAGE_FLAGS = 0x8220;
    static constexpr uint32_t GL_BUFFER_UPDATE_BARRIER_BIT = 0x00000200;
    static constexpr uint32_t GL_BUFFER_USAGE = 0x8765;
    static constexpr uint32_t GL_BUFFER_VARIABLE = 0x92E5;
    static constexpr uint32_t GL_BYTE = 0x1400;
    static constexpr uint32_t GL_C3F_V3F = 0x2A24;
    static constexpr uint32_t GL_C4F_N3F_V3F = 0x2A26;
    static constexpr uint32_t GL_C4UB_V2F = 0x2A22;
    static constexpr uint32_t GL_C4UB_V3F = 0x2A23;
    static constexpr uint32_t GL_CAVEAT_SUPPORT = 0x82B8;
    static constexpr uint32_t GL_CCW = 0x0901;
    static constexpr uint32_t GL_CLAMP = 0x2900;
    static constexpr uint32_t GL_CLAMP_FRAGMENT_COLOR = 0x891B;
    static constexpr uint32_t GL_CLAMP_READ_COLOR = 0x891C;
    static constexpr uint32_t GL_CLAMP_TO_BORDER = 0x812D;
    static constexpr uint32_t GL_CLAMP_TO_EDGE = 0x812F;
    static constexpr uint32_t GL_CLAMP_VERTEX_COLOR = 0x891A;
    static constexpr uint32_t GL_CLEAR = 0x1500;
    static constexpr uint32_t GL_CLEAR_BUFFER = 0x82B4;
    static constexpr uint32_t GL_CLEAR_TEXTURE = 0x9365;
    static constexpr uint32_t GL_CLIENT_ACTIVE_TEXTURE = 0x84E1;
    static constexpr uint32_t GL_CLIENT_ALL_ATTRIB_BITS = 0xFFFFFFFF;
    static constexpr uint32_t GL_CLIENT_ATTRIB_STACK_DEPTH = 0x0BB1;
    static constexpr uint32_t GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT = 0x00004000;
    static constexpr uint32_t GL_CLIENT_PIXEL_STORE_BIT = 0x00000001;
    static constexpr uint32_t GL_CLIENT_STORAGE_BIT = 0x0200;
    static constexpr uint32_t GL_CLIENT_VERTEX_ARRAY_BIT = 0x00000002;
    static constexpr uint32_t GL_CLIPPING_INPUT_PRIMITIVES = 0x82F6;
    static constexpr uint32_t GL_CLIPPING_OUTPUT_PRIMITIVES = 0x82F7;
    static constexpr uint32_t GL_CLIP_DEPTH_MODE = 0x935D;
    static constexpr uint32_t GL_CLIP_DISTANCE0 = 0x3000;
    static constexpr uint32_t GL_CLIP_DISTANCE1 = 0x3001;
    static constexpr uint32_t GL_CLIP_DISTANCE2 = 0x3002;
    static constexpr uint32_t GL_CLIP_DISTANCE3 = 0x3003;
    static constexpr uint32_t GL_CLIP_DISTANCE4 = 0x3004;
    static constexpr uint32_t GL_CLIP_DISTANCE5 = 0x3005;
    static constexpr uint32_t GL_CLIP_DISTANCE6 = 0x3006;
    static constexpr uint32_t GL_CLIP_DISTANCE7 = 0x3007;
    static constexpr uint32_t GL_CLIP_ORIGIN = 0x935C;
    static constexpr uint32_t GL_CLIP_PLANE0 = 0x3000;
    static constexpr uint32_t GL_CLIP_PLANE1 = 0x3001;
    static constexpr uint32_t GL_CLIP_PLANE2 = 0x3002;
    static constexpr uint32_t GL_CLIP_PLANE3 = 0x3003;
    static constexpr uint32_t GL_CLIP_PLANE4 = 0x3004;
    static constexpr uint32_t GL_CLIP_PLANE5 = 0x3005;
    static constexpr uint32_t GL_COEFF = 0x0A00;
    static constexpr uint32_t GL_COLOR = 0x1800;
    static constexpr uint32_t GL_COLOR_ARRAY = 0x8076;
    static constexpr uint32_t GL_COLOR_ARRAY_BUFFER_BINDING = 0x8898;
    static constexpr uint32_t GL_COLOR_ARRAY_POINTER = 0x8090;
    static constexpr uint32_t GL_COLOR_ARRAY_SIZE = 0x8081;
    static constexpr uint32_t GL_COLOR_ARRAY_STRIDE = 0x8083;
    static constexpr uint32_t GL_COLOR_ARRAY_TYPE = 0x8082;
    static constexpr uint32_t GL_COLOR_ATTACHMENT0 = 0x8CE0;
    static constexpr uint32_t GL_COLOR_ATTACHMENT1 = 0x8CE1;
    static constexpr uint32_t GL_COLOR_ATTACHMENT10 = 0x8CEA;
    static constexpr uint32_t GL_COLOR_ATTACHMENT11 = 0x8CEB;
    static constexpr uint32_t GL_COLOR_ATTACHMENT12 = 0x8CEC;
    static constexpr uint32_t GL_COLOR_ATTACHMENT13 = 0x8CED;
    static constexpr uint32_t GL_COLOR_ATTACHMENT14 = 0x8CEE;
    static constexpr uint32_t GL_COLOR_ATTACHMENT15 = 0x8CEF;
    static constexpr uint32_t GL_COLOR_ATTACHMENT16 = 0x8CF0;
    static constexpr uint32_t GL_COLOR_ATTACHMENT17 = 0x8CF1;
    static constexpr uint32_t GL_COLOR_ATTACHMENT18 = 0x8CF2;
    static constexpr uint32_t GL_COLOR_ATTACHMENT19 = 0x8CF3;
    static constexpr uint32_t GL_COLOR_ATTACHMENT2 = 0x8CE2;
    static constexpr uint32_t GL_COLOR_ATTACHMENT20 = 0x8CF4;
    static constexpr uint32_t GL_COLOR_ATTACHMENT21 = 0x8CF5;
    static constexpr uint32_t GL_COLOR_ATTACHMENT22 = 0x8CF6;
    static constexpr uint32_t GL_COLOR_ATTACHMENT23 = 0x8CF7;
    static constexpr uint32_t GL_COLOR_ATTACHMENT24 = 0x8CF8;
    static constexpr uint32_t GL_COLOR_ATTACHMENT25 = 0x8CF9;
    static constexpr uint32_t GL_COLOR_ATTACHMENT26 = 0x8CFA;
    static constexpr uint32_t GL_COLOR_ATTACHMENT27 = 0x8CFB;
    static constexpr uint32_t GL_COLOR_ATTACHMENT28 = 0x8CFC;
    static constexpr uint32_t GL_COLOR_ATTACHMENT29 = 0x8CFD;
    static constexpr uint32_t GL_COLOR_ATTACHMENT3 = 0x8CE3;
    static constexpr uint32_t GL_COLOR_ATTACHMENT30 = 0x8CFE;
    static constexpr uint32_t GL_COLOR_ATTACHMENT31 = 0x8CFF;
    static constexpr uint32_t GL_COLOR_ATTACHMENT4 = 0x8CE4;
    static constexpr uint32_t GL_COLOR_ATTACHMENT5 = 0x8CE5;
    static constexpr uint32_t GL_COLOR_ATTACHMENT6 = 0x8CE6;
    static constexpr uint32_t GL_COLOR_ATTACHMENT7 = 0x8CE7;
    static constexpr uint32_t GL_COLOR_ATTACHMENT8 = 0x8CE8;
    static constexpr uint32_t GL_COLOR_ATTACHMENT9 = 0x8CE9;
    static constexpr uint32_t GL_COLOR_BUFFER_BIT = 0x00004000;
    static constexpr uint32_t GL_COLOR_CLEAR_VALUE = 0x0C22;
    static constexpr uint32_t GL_COLOR_COMPONENTS = 0x8283;
    static constexpr uint32_t GL_COLOR_ENCODING = 0x8296;
    static constexpr uint32_t GL_COLOR_INDEX = 0x1900;
    static constexpr uint32_t GL_COLOR_INDEXES = 0x1603;
    static constexpr uint32_t GL_COLOR_LOGIC_OP = 0x0BF2;
    static constexpr uint32_t GL_COLOR_MATERIAL = 0x0B57;
    static constexpr uint32_t GL_COLOR_MATERIAL_FACE = 0x0B55;
    static constexpr uint32_t GL_COLOR_MATERIAL_PARAMETER = 0x0B56;
    static constexpr uint32_t GL_COLOR_RENDERABLE = 0x8286;
    static constexpr uint32_t GL_COLOR_SUM = 0x8458;
    static constexpr uint32_t GL_COLOR_TABLE = 0x80D0;
    static constexpr uint32_t GL_COLOR_WRITEMASK = 0x0C23;
    static constexpr uint32_t GL_COMBINE = 0x8570;
    static constexpr uint32_t GL_COMBINE_ALPHA = 0x8572;
    static constexpr uint32_t GL_COMBINE_RGB = 0x8571;
    static constexpr uint32_t GL_COMMAND_BARRIER_BIT = 0x00000040;
    static constexpr uint32_t GL_COMPARE_REF_TO_TEXTURE = 0x884E;
    static constexpr uint32_t GL_COMPARE_R_TO_TEXTURE = 0x884E;
    static constexpr uint32_t GL_COMPATIBLE_SUBROUTINES = 0x8E4B;
    static constexpr uint32_t GL_COMPILE = 0x1300;
    static constexpr uint32_t GL_COMPILE_AND_EXECUTE = 0x1301;
    static constexpr uint32_t GL_COMPILE_STATUS = 0x8B81;
    static constexpr uint32_t GL_COMPRESSED_ALPHA = 0x84E9;
    static constexpr uint32_t GL_COMPRESSED_INTENSITY = 0x84EC;
    static constexpr uint32_t GL_COMPRESSED_LUMINANCE = 0x84EA;
    static constexpr uint32_t GL_COMPRESSED_LUMINANCE_ALPHA = 0x84EB;
    static constexpr uint32_t GL_COMPRESSED_R11_EAC = 0x9270;
    static constexpr uint32_t GL_COMPRESSED_RED = 0x8225;
    static constexpr uint32_t GL_COMPRESSED_RED_RGTC1 = 0x8DBB;
    static constexpr uint32_t GL_COMPRESSED_RG = 0x8226;
    static constexpr uint32_t GL_COMPRESSED_RG11_EAC = 0x9272;
    static constexpr uint32_t GL_COMPRESSED_RGB = 0x84ED;
    static constexpr uint32_t GL_COMPRESSED_RGB8_ETC2 = 0x9274;
    static constexpr uint32_t GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 = 0x9276;
    static constexpr uint32_t GL_COMPRESSED_RGBA = 0x84EE;
    static constexpr uint32_t GL_COMPRESSED_RGBA8_ETC2_EAC = 0x9278;
    static constexpr uint32_t GL_COMPRESSED_RGBA_BPTC_UNORM = 0x8E8C;
    static constexpr uint32_t GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT = 0x8E8E;
    static constexpr uint32_t GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT = 0x8E8F;
    static constexpr uint32_t GL_COMPRESSED_RG_RGTC2 = 0x8DBD;
    static constexpr uint32_t GL_COMPRESSED_SIGNED_R11_EAC = 0x9271;
    static constexpr uint32_t GL_COMPRESSED_SIGNED_RED_RGTC1 = 0x8DBC;
    static constexpr uint32_t GL_COMPRESSED_SIGNED_RG11_EAC = 0x9273;
    static constexpr uint32_t GL_COMPRESSED_SIGNED_RG_RGTC2 = 0x8DBE;
    static constexpr uint32_t GL_COMPRESSED_SLUMINANCE = 0x8C4A;
    static constexpr uint32_t GL_COMPRESSED_SLUMINANCE_ALPHA = 0x8C4B;
    static constexpr uint32_t GL_COMPRESSED_SRGB = 0x8C48;
    static constexpr uint32_t GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC = 0x9279;
    static constexpr uint32_t GL_COMPRESSED_SRGB8_ETC2 = 0x9275;
    static constexpr uint32_t GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 = 0x9277;
    static constexpr uint32_t GL_COMPRESSED_SRGB_ALPHA = 0x8C49;
    static constexpr uint32_t GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM = 0x8E8D;
    static constexpr uint32_t GL_COMPRESSED_TEXTURE_FORMATS = 0x86A3;
    static constexpr uint32_t GL_COMPUTE_SHADER = 0x91B9;
    static constexpr uint32_t GL_COMPUTE_SHADER_BIT = 0x00000020;
    static constexpr uint32_t GL_COMPUTE_SHADER_INVOCATIONS = 0x82F5;
    static constexpr uint32_t GL_COMPUTE_SUBROUTINE = 0x92ED;
    static constexpr uint32_t GL_COMPUTE_SUBROUTINE_UNIFORM = 0x92F3;
    static constexpr uint32_t GL_COMPUTE_TEXTURE = 0x82A0;
    static constexpr uint32_t GL_COMPUTE_WORK_GROUP_SIZE = 0x8267;
    static constexpr uint32_t GL_CONDITION_SATISFIED = 0x911C;
    static constexpr uint32_t GL_CONSTANT = 0x8576;
    static constexpr uint32_t GL_CONSTANT_ALPHA = 0x8003;
    static constexpr uint32_t GL_CONSTANT_ATTENUATION = 0x1207;
    static constexpr uint32_t GL_CONSTANT_COLOR = 0x8001;
    static constexpr uint32_t GL_CONTEXT_COMPATIBILITY_PROFILE_BIT = 0x00000002;
    static constexpr uint32_t GL_CONTEXT_CORE_PROFILE_BIT = 0x00000001;
    static constexpr uint32_t GL_CONTEXT_FLAGS = 0x821E;
    static constexpr uint32_t GL_CONTEXT_FLAG_DEBUG_BIT = 0x00000002;
    static constexpr uint32_t GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT = 0x00000001;
    static constexpr uint32_t GL_CONTEXT_FLAG_NO_ERROR_BIT = 0x00000008;
    static constexpr uint32_t GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT = 0x00000004;
    static constexpr uint32_t GL_CONTEXT_LOST = 0x0507;
    static constexpr uint32_t GL_CONTEXT_PROFILE_MASK = 0x9126;
    static constexpr uint32_t GL_CONTEXT_RELEASE_BEHAVIOR = 0x82FB;
    static constexpr uint32_t GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH = 0x82FC;
    static constexpr uint32_t GL_CONVOLUTION_1D = 0x8010;
    static constexpr uint32_t GL_CONVOLUTION_2D = 0x8011;
    static constexpr uint32_t GL_COORD_REPLACE = 0x8862;
    static constexpr uint32_t GL_COPY = 0x1503;
    static constexpr uint32_t GL_COPY_INVERTED = 0x150C;
    static constexpr uint32_t GL_COPY_PIXEL_TOKEN = 0x0706;
    static constexpr uint32_t GL_COPY_READ_BUFFER = 0x8F36;
    static constexpr uint32_t GL_COPY_READ_BUFFER_BINDING = 0x8F36;
    static constexpr uint32_t GL_COPY_WRITE_BUFFER = 0x8F37;
    static constexpr uint32_t GL_COPY_WRITE_BUFFER_BINDING = 0x8F37;
    static constexpr uint32_t GL_CULL_FACE = 0x0B44;
    static constexpr uint32_t GL_CULL_FACE_MODE = 0x0B45;
    static constexpr uint32_t GL_CURRENT_BIT = 0x00000001;
    static constexpr uint32_t GL_CURRENT_COLOR = 0x0B00;
    static constexpr uint32_t GL_CURRENT_FOG_COORD = 0x8453;
    static constexpr uint32_t GL_CURRENT_FOG_COORDINATE = 0x8453;
    static constexpr uint32_t GL_CURRENT_INDEX = 0x0B01;
    static constexpr uint32_t GL_CURRENT_NORMAL = 0x0B02;
    static constexpr uint32_t GL_CURRENT_PROGRAM = 0x8B8D;
    static constexpr uint32_t GL_CURRENT_QUERY = 0x8865;
    static constexpr uint32_t GL_CURRENT_RASTER_COLOR = 0x0B04;
    static constexpr uint32_t GL_CURRENT_RASTER_DISTANCE = 0x0B09;
    static constexpr uint32_t GL_CURRENT_RASTER_INDEX = 0x0B05;
    static constexpr uint32_t GL_CURRENT_RASTER_POSITION = 0x0B07;
    static constexpr uint32_t GL_CURRENT_RASTER_POSITION_VALID = 0x0B08;
    static constexpr uint32_t GL_CURRENT_RASTER_SECONDARY_COLOR = 0x845F;
    static constexpr uint32_t GL_CURRENT_RASTER_TEXTURE_COORDS = 0x0B06;
    static constexpr uint32_t GL_CURRENT_SECONDARY_COLOR = 0x8459;
    static constexpr uint32_t GL_CURRENT_TEXTURE_COORDS = 0x0B03;
    static constexpr uint32_t GL_CURRENT_VERTEX_ATTRIB = 0x8626;
    static constexpr uint32_t GL_CW = 0x0900;
    static constexpr uint32_t GL_DEBUG_CALLBACK_FUNCTION = 0x8244;
    static constexpr uint32_t GL_DEBUG_CALLBACK_USER_PARAM = 0x8245;
    static constexpr uint32_t GL_DEBUG_GROUP_STACK_DEPTH = 0x826D;
    static constexpr uint32_t GL_DEBUG_LOGGED_MESSAGES = 0x9145;
    static constexpr uint32_t GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH = 0x8243;
    static constexpr uint32_t GL_DEBUG_OUTPUT = 0x92E0;
    static constexpr uint32_t GL_DEBUG_OUTPUT_SYNCHRONOUS = 0x8242;
    static constexpr uint32_t GL_DEBUG_SEVERITY_HIGH = 0x9146;
    static constexpr uint32_t GL_DEBUG_SEVERITY_LOW = 0x9148;
    static constexpr uint32_t GL_DEBUG_SEVERITY_MEDIUM = 0x9147;
    static constexpr uint32_t GL_DEBUG_SEVERITY_NOTIFICATION = 0x826B;
    static constexpr uint32_t GL_DEBUG_SOURCE_API = 0x8246;
    static constexpr uint32_t GL_DEBUG_SOURCE_APPLICATION = 0x824A;
    static constexpr uint32_t GL_DEBUG_SOURCE_OTHER = 0x824B;
    static constexpr uint32_t GL_DEBUG_SOURCE_SHADER_COMPILER = 0x8248;
    static constexpr uint32_t GL_DEBUG_SOURCE_THIRD_PARTY = 0x8249;
    static constexpr uint32_t GL_DEBUG_SOURCE_WINDOW_SYSTEM = 0x8247;
    static constexpr uint32_t GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR = 0x824D;
    static constexpr uint32_t GL_DEBUG_TYPE_ERROR = 0x824C;
    static constexpr uint32_t GL_DEBUG_TYPE_MARKER = 0x8268;
    static constexpr uint32_t GL_DEBUG_TYPE_OTHER = 0x8251;
    static constexpr uint32_t GL_DEBUG_TYPE_PERFORMANCE = 0x8250;
    static constexpr uint32_t GL_DEBUG_TYPE_POP_GROUP = 0x826A;
    static constexpr uint32_t GL_DEBUG_TYPE_PORTABILITY = 0x824F;
    static constexpr uint32_t GL_DEBUG_TYPE_PUSH_GROUP = 0x8269;
    static constexpr uint32_t GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR = 0x824E;
    static constexpr uint32_t GL_DECAL = 0x2101;
    static constexpr uint32_t GL_DECR = 0x1E03;
    static constexpr uint32_t GL_DECR_WRAP = 0x8508;
    static constexpr uint32_t GL_DELETE_STATUS = 0x8B80;
    static constexpr uint32_t GL_DEPTH = 0x1801;
    static constexpr uint32_t GL_DEPTH24_STENCIL8 = 0x88F0;
    static constexpr uint32_t GL_DEPTH32F_STENCIL8 = 0x8CAD;
    static constexpr uint32_t GL_DEPTH_ATTACHMENT = 0x8D00;
    static constexpr uint32_t GL_DEPTH_BIAS = 0x0D1F;
    static constexpr uint32_t GL_DEPTH_BITS = 0x0D56;
    static constexpr uint32_t GL_DEPTH_BUFFER_BIT = 0x00000100;
    static constexpr uint32_t GL_DEPTH_CLAMP = 0x864F;
    static constexpr uint32_t GL_DEPTH_CLEAR_VALUE = 0x0B73;
    static constexpr uint32_t GL_DEPTH_COMPONENT = 0x1902;
    static constexpr uint32_t GL_DEPTH_COMPONENT16 = 0x81A5;
    static constexpr uint32_t GL_DEPTH_COMPONENT24 = 0x81A6;
    static constexpr uint32_t GL_DEPTH_COMPONENT32 = 0x81A7;
    static constexpr uint32_t GL_DEPTH_COMPONENT32F = 0x8CAC;
    static constexpr uint32_t GL_DEPTH_COMPONENTS = 0x8284;
    static constexpr uint32_t GL_DEPTH_FUNC = 0x0B74;
    static constexpr uint32_t GL_DEPTH_RANGE = 0x0B70;
    static constexpr uint32_t GL_DEPTH_RENDERABLE = 0x8287;
    static constexpr uint32_t GL_DEPTH_SCALE = 0x0D1E;
    static constexpr uint32_t GL_DEPTH_STENCIL = 0x84F9;
    static constexpr uint32_t GL_DEPTH_STENCIL_ATTACHMENT = 0x821A;
    static constexpr uint32_t GL_DEPTH_STENCIL_TEXTURE_MODE = 0x90EA;
    static constexpr uint32_t GL_DEPTH_TEST = 0x0B71;
    static constexpr uint32_t GL_DEPTH_TEXTURE_MODE = 0x884B;
    static constexpr uint32_t GL_DEPTH_WRITEMASK = 0x0B72;
    static constexpr uint32_t GL_DIFFUSE = 0x1201;
    static constexpr uint32_t GL_DISPATCH_INDIRECT_BUFFER = 0x90EE;
    static constexpr uint32_t GL_DISPATCH_INDIRECT_BUFFER_BINDING = 0x90EF;
    static constexpr uint32_t GL_DISPLAY_LIST = 0x82E7;
    static constexpr uint32_t GL_DITHER = 0x0BD0;
    static constexpr uint32_t GL_DONT_CARE = 0x1100;
    static constexpr uint32_t GL_DOT3_RGB = 0x86AE;
    static constexpr uint32_t GL_DOT3_RGBA = 0x86AF;
    static constexpr uint32_t GL_DOUBLE = 0x140A;
    static constexpr uint32_t GL_DOUBLEBUFFER = 0x0C32;
    static constexpr uint32_t GL_DOUBLE_MAT2 = 0x8F46;
    static constexpr uint32_t GL_DOUBLE_MAT2x3 = 0x8F49;
    static constexpr uint32_t GL_DOUBLE_MAT2x4 = 0x8F4A;
    static constexpr uint32_t GL_DOUBLE_MAT3 = 0x8F47;
    static constexpr uint32_t GL_DOUBLE_MAT3x2 = 0x8F4B;
    static constexpr uint32_t GL_DOUBLE_MAT3x4 = 0x8F4C;
    static constexpr uint32_t GL_DOUBLE_MAT4 = 0x8F48;
    static constexpr uint32_t GL_DOUBLE_MAT4x2 = 0x8F4D;
    static constexpr uint32_t GL_DOUBLE_MAT4x3 = 0x8F4E;
    static constexpr uint32_t GL_DOUBLE_VEC2 = 0x8FFC;
    static constexpr uint32_t GL_DOUBLE_VEC3 = 0x8FFD;
    static constexpr uint32_t GL_DOUBLE_VEC4 = 0x8FFE;
    static constexpr uint32_t GL_DRAW_BUFFER = 0x0C01;
    static constexpr uint32_t GL_DRAW_BUFFER0 = 0x8825;
    static constexpr uint32_t GL_DRAW_BUFFER1 = 0x8826;
    static constexpr uint32_t GL_DRAW_BUFFER10 = 0x882F;
    static constexpr uint32_t GL_DRAW_BUFFER11 = 0x8830;
    static constexpr uint32_t GL_DRAW_BUFFER12 = 0x8831;
    static constexpr uint32_t GL_DRAW_BUFFER13 = 0x8832;
    static constexpr uint32_t GL_DRAW_BUFFER14 = 0x8833;
    static constexpr uint32_t GL_DRAW_BUFFER15 = 0x8834;
    static constexpr uint32_t GL_DRAW_BUFFER2 = 0x8827;
    static constexpr uint32_t GL_DRAW_BUFFER3 = 0x8828;
    static constexpr uint32_t GL_DRAW_BUFFER4 = 0x8829;
    static constexpr uint32_t GL_DRAW_BUFFER5 = 0x882A;
    static constexpr uint32_t GL_DRAW_BUFFER6 = 0x882B;
    static constexpr uint32_t GL_DRAW_BUFFER7 = 0x882C;
    static constexpr uint32_t GL_DRAW_BUFFER8 = 0x882D;
    static constexpr uint32_t GL_DRAW_BUFFER9 = 0x882E;
    static constexpr uint32_t GL_DRAW_FRAMEBUFFER = 0x8CA9;
    static constexpr uint32_t GL_DRAW_FRAMEBUFFER_BINDING = 0x8CA6;
    static constexpr uint32_t GL_DRAW_INDIRECT_BUFFER = 0x8F3F;
    static constexpr uint32_t GL_DRAW_INDIRECT_BUFFER_BINDING = 0x8F43;
    static constexpr uint32_t GL_DRAW_PIXEL_TOKEN = 0x0705;
    static constexpr uint32_t GL_DST_ALPHA = 0x0304;
    static constexpr uint32_t GL_DST_COLOR = 0x0306;
    static constexpr uint32_t GL_DYNAMIC_COPY = 0x88EA;
    static constexpr uint32_t GL_DYNAMIC_DRAW = 0x88E8;
    static constexpr uint32_t GL_DYNAMIC_READ = 0x88E9;
    static constexpr uint32_t GL_DYNAMIC_STORAGE_BIT = 0x0100;
    static constexpr uint32_t GL_EDGE_FLAG = 0x0B43;
    static constexpr uint32_t GL_EDGE_FLAG_ARRAY = 0x8079;
    static constexpr uint32_t GL_EDGE_FLAG_ARRAY_BUFFER_BINDING = 0x889B;
    static constexpr uint32_t GL_EDGE_FLAG_ARRAY_POINTER = 0x8093;
    static constexpr uint32_t GL_EDGE_FLAG_ARRAY_STRIDE = 0x808C;
    static constexpr uint32_t GL_ELEMENT_ARRAY_BARRIER_BIT = 0x00000002;
    static constexpr uint32_t GL_ELEMENT_ARRAY_BUFFER = 0x8893;
    static constexpr uint32_t GL_ELEMENT_ARRAY_BUFFER_BINDING = 0x8895;
    static constexpr uint32_t GL_EMISSION = 0x1600;
    static constexpr uint32_t GL_ENABLE_BIT = 0x00002000;
    static constexpr uint32_t GL_EQUAL = 0x0202;
    static constexpr uint32_t GL_EQUIV = 0x1509;
    static constexpr uint32_t GL_EVAL_BIT = 0x00010000;
    static constexpr uint32_t GL_EXP = 0x0800;
    static constexpr uint32_t GL_EXP2 = 0x0801;
    static constexpr uint32_t GL_EXTENSIONS = 0x1F03;
    static constexpr uint32_t GL_EYE_LINEAR = 0x2400;
    static constexpr uint32_t GL_EYE_PLANE = 0x2502;
    static constexpr uint32_t GL_FALSE = 0;
    static constexpr uint32_t GL_FASTEST = 0x1101;
    static constexpr uint32_t GL_FEEDBACK = 0x1C01;
    static constexpr uint32_t GL_FEEDBACK_BUFFER_POINTER = 0x0DF0;
    static constexpr uint32_t GL_FEEDBACK_BUFFER_SIZE = 0x0DF1;
    static constexpr uint32_t GL_FEEDBACK_BUFFER_TYPE = 0x0DF2;
    static constexpr uint32_t GL_FILL = 0x1B02;
    static constexpr uint32_t GL_FILTER = 0x829A;
    static constexpr uint32_t GL_FIRST_VERTEX_CONVENTION = 0x8E4D;
    static constexpr uint32_t GL_FIXED = 0x140C;
    static constexpr uint32_t GL_FIXED_ONLY = 0x891D;
    static constexpr uint32_t GL_FLAT = 0x1D00;
    static constexpr uint32_t GL_FLOAT = 0x1406;
    static constexpr uint32_t GL_FLOAT_32_UNSIGNED_INT_24_8_REV = 0x8DAD;
    static constexpr uint32_t GL_FLOAT_MAT2 = 0x8B5A;
    static constexpr uint32_t GL_FLOAT_MAT2x3 = 0x8B65;
    static constexpr uint32_t GL_FLOAT_MAT2x4 = 0x8B66;
    static constexpr uint32_t GL_FLOAT_MAT3 = 0x8B5B;
    static constexpr uint32_t GL_FLOAT_MAT3x2 = 0x8B67;
    static constexpr uint32_t GL_FLOAT_MAT3x4 = 0x8B68;
    static constexpr uint32_t GL_FLOAT_MAT4 = 0x8B5C;
    static constexpr uint32_t GL_FLOAT_MAT4x2 = 0x8B69;
    static constexpr uint32_t GL_FLOAT_MAT4x3 = 0x8B6A;
    static constexpr uint32_t GL_FLOAT_VEC2 = 0x8B50;
    static constexpr uint32_t GL_FLOAT_VEC3 = 0x8B51;
    static constexpr uint32_t GL_FLOAT_VEC4 = 0x8B52;
    static constexpr uint32_t GL_FOG = 0x0B60;
    static constexpr uint32_t GL_FOG_BIT = 0x00000080;
    static constexpr uint32_t GL_FOG_COLOR = 0x0B66;
    static constexpr uint32_t GL_FOG_COORD = 0x8451;
    static constexpr uint32_t GL_FOG_COORDINATE = 0x8451;
    static constexpr uint32_t GL_FOG_COORDINATE_ARRAY = 0x8457;
    static constexpr uint32_t GL_FOG_COORDINATE_ARRAY_BUFFER_BINDING = 0x889D;
    static constexpr uint32_t GL_FOG_COORDINATE_ARRAY_POINTER = 0x8456;
    static constexpr uint32_t GL_FOG_COORDINATE_ARRAY_STRIDE = 0x8455;
    static constexpr uint32_t GL_FOG_COORDINATE_ARRAY_TYPE = 0x8454;
    static constexpr uint32_t GL_FOG_COORDINATE_SOURCE = 0x8450;
    static constexpr uint32_t GL_FOG_COORD_ARRAY = 0x8457;
    static constexpr uint32_t GL_FOG_COORD_ARRAY_BUFFER_BINDING = 0x889D;
    static constexpr uint32_t GL_FOG_COORD_ARRAY_POINTER = 0x8456;
    static constexpr uint32_t GL_FOG_COORD_ARRAY_STRIDE = 0x8455;
    static constexpr uint32_t GL_FOG_COORD_ARRAY_TYPE = 0x8454;
    static constexpr uint32_t GL_FOG_COORD_SRC = 0x8450;
    static constexpr uint32_t GL_FOG_DENSITY = 0x0B62;
    static constexpr uint32_t GL_FOG_END = 0x0B64;
    static constexpr uint32_t GL_FOG_HINT = 0x0C54;
    static constexpr uint32_t GL_FOG_INDEX = 0x0B61;
    static constexpr uint32_t GL_FOG_MODE = 0x0B65;
    static constexpr uint32_t GL_FOG_START = 0x0B63;
    static constexpr uint32_t GL_FRACTIONAL_EVEN = 0x8E7C;
    static constexpr uint32_t GL_FRACTIONAL_ODD = 0x8E7B;
    static constexpr uint32_t GL_FRAGMENT_DEPTH = 0x8452;
    static constexpr uint32_t GL_FRAGMENT_INTERPOLATION_OFFSET_BITS = 0x8E5D;
    static constexpr uint32_t GL_FRAGMENT_SHADER = 0x8B30;
    static constexpr uint32_t GL_FRAGMENT_SHADER_BIT = 0x00000002;
    static constexpr uint32_t GL_FRAGMENT_SHADER_DERIVATIVE_HINT = 0x8B8B;
    static constexpr uint32_t GL_FRAGMENT_SHADER_INVOCATIONS = 0x82F4;
    static constexpr uint32_t GL_FRAGMENT_SUBROUTINE = 0x92EC;
    static constexpr uint32_t GL_FRAGMENT_SUBROUTINE_UNIFORM = 0x92F2;
    static constexpr uint32_t GL_FRAGMENT_TEXTURE = 0x829F;
    static constexpr uint32_t GL_FRAMEBUFFER = 0x8D40;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE = 0x8215;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE = 0x8214;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING = 0x8210;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE = 0x8211;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE = 0x8216;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE = 0x8213;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_LAYERED = 0x8DA7;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME = 0x8CD1;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE = 0x8CD0;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE = 0x8212;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE = 0x8217;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE = 0x8CD3;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER = 0x8CD4;
    static constexpr uint32_t GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL = 0x8CD2;
    static constexpr uint32_t GL_FRAMEBUFFER_BARRIER_BIT = 0x00000400;
    static constexpr uint32_t GL_FRAMEBUFFER_BINDING = 0x8CA6;
    static constexpr uint32_t GL_FRAMEBUFFER_BLEND = 0x828B;
    static constexpr uint32_t GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
    static constexpr uint32_t GL_FRAMEBUFFER_DEFAULT = 0x8218;
    static constexpr uint32_t GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS = 0x9314;
    static constexpr uint32_t GL_FRAMEBUFFER_DEFAULT_HEIGHT = 0x9311;
    static constexpr uint32_t GL_FRAMEBUFFER_DEFAULT_LAYERS = 0x9312;
    static constexpr uint32_t GL_FRAMEBUFFER_DEFAULT_SAMPLES = 0x9313;
    static constexpr uint32_t GL_FRAMEBUFFER_DEFAULT_WIDTH = 0x9310;
    static constexpr uint32_t GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT = 0x8CD6;
    static constexpr uint32_t GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER = 0x8CDB;
    static constexpr uint32_t GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS = 0x8DA8;
    static constexpr uint32_t GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT = 0x8CD7;
    static constexpr uint32_t GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE = 0x8D56;
    static constexpr uint32_t GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER = 0x8CDC;
    static constexpr uint32_t GL_FRAMEBUFFER_RENDERABLE = 0x8289;
    static constexpr uint32_t GL_FRAMEBUFFER_RENDERABLE_LAYERED = 0x828A;
    static constexpr uint32_t GL_FRAMEBUFFER_SRGB = 0x8DB9;
    static constexpr uint32_t GL_FRAMEBUFFER_UNDEFINED = 0x8219;
    static constexpr uint32_t GL_FRAMEBUFFER_UNSUPPORTED = 0x8CDD;
    static constexpr uint32_t GL_FRONT = 0x0404;
    static constexpr uint32_t GL_FRONT_AND_BACK = 0x0408;
    static constexpr uint32_t GL_FRONT_FACE = 0x0B46;
    static constexpr uint32_t GL_FRONT_LEFT = 0x0400;
    static constexpr uint32_t GL_FRONT_RIGHT = 0x0401;
    static constexpr uint32_t GL_FULL_SUPPORT = 0x82B7;
    static constexpr uint32_t GL_FUNC_ADD = 0x8006;
    static constexpr uint32_t GL_FUNC_REVERSE_SUBTRACT = 0x800B;
    static constexpr uint32_t GL_FUNC_SUBTRACT = 0x800A;
    static constexpr uint32_t GL_GENERATE_MIPMAP = 0x8191;
    static constexpr uint32_t GL_GENERATE_MIPMAP_HINT = 0x8192;
    static constexpr uint32_t GL_GEOMETRY_INPUT_TYPE = 0x8917;
    static constexpr uint32_t GL_GEOMETRY_OUTPUT_TYPE = 0x8918;
    static constexpr uint32_t GL_GEOMETRY_SHADER = 0x8DD9;
    static constexpr uint32_t GL_GEOMETRY_SHADER_BIT = 0x00000004;
    static constexpr uint32_t GL_GEOMETRY_SHADER_INVOCATIONS = 0x887F;
    static constexpr uint32_t GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED = 0x82F3;
    static constexpr uint32_t GL_GEOMETRY_SUBROUTINE = 0x92EB;
    static constexpr uint32_t GL_GEOMETRY_SUBROUTINE_UNIFORM = 0x92F1;
    static constexpr uint32_t GL_GEOMETRY_TEXTURE = 0x829E;
    static constexpr uint32_t GL_GEOMETRY_VERTICES_OUT = 0x8916;
    static constexpr uint32_t GL_GEQUAL = 0x0206;
    static constexpr uint32_t GL_GET_TEXTURE_IMAGE_FORMAT = 0x8291;
    static constexpr uint32_t GL_GET_TEXTURE_IMAGE_TYPE = 0x8292;
    static constexpr uint32_t GL_GREATER = 0x0204;
    static constexpr uint32_t GL_GREEN = 0x1904;
    static constexpr uint32_t GL_GREEN_BIAS = 0x0D19;
    static constexpr uint32_t GL_GREEN_BITS = 0x0D53;
    static constexpr uint32_t GL_GREEN_INTEGER = 0x8D95;
    static constexpr uint32_t GL_GREEN_SCALE = 0x0D18;
    static constexpr uint32_t GL_GUILTY_CONTEXT_RESET = 0x8253;
    static constexpr uint32_t GL_HALF_FLOAT = 0x140B;
    static constexpr uint32_t GL_HIGH_FLOAT = 0x8DF2;
    static constexpr uint32_t GL_HIGH_INT = 0x8DF5;
    static constexpr uint32_t GL_HINT_BIT = 0x00008000;
    static constexpr uint32_t GL_HISTOGRAM = 0x8024;
    static constexpr uint32_t GL_IMAGE_1D = 0x904C;
    static constexpr uint32_t GL_IMAGE_1D_ARRAY = 0x9052;
    static constexpr uint32_t GL_IMAGE_2D = 0x904D;
    static constexpr uint32_t GL_IMAGE_2D_ARRAY = 0x9053;
    static constexpr uint32_t GL_IMAGE_2D_MULTISAMPLE = 0x9055;
    static constexpr uint32_t GL_IMAGE_2D_MULTISAMPLE_ARRAY = 0x9056;
    static constexpr uint32_t GL_IMAGE_2D_RECT = 0x904F;
    static constexpr uint32_t GL_IMAGE_3D = 0x904E;
    static constexpr uint32_t GL_IMAGE_BINDING_ACCESS = 0x8F3E;
    static constexpr uint32_t GL_IMAGE_BINDING_FORMAT = 0x906E;
    static constexpr uint32_t GL_IMAGE_BINDING_LAYER = 0x8F3D;
    static constexpr uint32_t GL_IMAGE_BINDING_LAYERED = 0x8F3C;
    static constexpr uint32_t GL_IMAGE_BINDING_LEVEL = 0x8F3B;
    static constexpr uint32_t GL_IMAGE_BINDING_NAME = 0x8F3A;
    static constexpr uint32_t GL_IMAGE_BUFFER = 0x9051;
    static constexpr uint32_t GL_IMAGE_CLASS_10_10_10_2 = 0x82C3;
    static constexpr uint32_t GL_IMAGE_CLASS_11_11_10 = 0x82C2;
    static constexpr uint32_t GL_IMAGE_CLASS_1_X_16 = 0x82BE;
    static constexpr uint32_t GL_IMAGE_CLASS_1_X_32 = 0x82BB;
    static constexpr uint32_t GL_IMAGE_CLASS_1_X_8 = 0x82C1;
    static constexpr uint32_t GL_IMAGE_CLASS_2_X_16 = 0x82BD;
    static constexpr uint32_t GL_IMAGE_CLASS_2_X_32 = 0x82BA;
    static constexpr uint32_t GL_IMAGE_CLASS_2_X_8 = 0x82C0;
    static constexpr uint32_t GL_IMAGE_CLASS_4_X_16 = 0x82BC;
    static constexpr uint32_t GL_IMAGE_CLASS_4_X_32 = 0x82B9;
    static constexpr uint32_t GL_IMAGE_CLASS_4_X_8 = 0x82BF;
    static constexpr uint32_t GL_IMAGE_COMPATIBILITY_CLASS = 0x82A8;
    static constexpr uint32_t GL_IMAGE_CUBE = 0x9050;
    static constexpr uint32_t GL_IMAGE_CUBE_MAP_ARRAY = 0x9054;
    static constexpr uint32_t GL_IMAGE_FORMAT_COMPATIBILITY_BY_CLASS = 0x90C9;
    static constexpr uint32_t GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE = 0x90C8;
    static constexpr uint32_t GL_IMAGE_FORMAT_COMPATIBILITY_TYPE = 0x90C7;
    static constexpr uint32_t GL_IMAGE_PIXEL_FORMAT = 0x82A9;
    static constexpr uint32_t GL_IMAGE_PIXEL_TYPE = 0x82AA;
    static constexpr uint32_t GL_IMAGE_TEXEL_SIZE = 0x82A7;
    static constexpr uint32_t GL_IMPLEMENTATION_COLOR_READ_FORMAT = 0x8B9B;
    static constexpr uint32_t GL_IMPLEMENTATION_COLOR_READ_TYPE = 0x8B9A;
    static constexpr uint32_t GL_INCR = 0x1E02;
    static constexpr uint32_t GL_INCR_WRAP = 0x8507;
    static constexpr uint32_t GL_INDEX = 0x8222;
    static constexpr uint32_t GL_INDEX_ARRAY = 0x8077;
    static constexpr uint32_t GL_INDEX_ARRAY_BUFFER_BINDING = 0x8899;
    static constexpr uint32_t GL_INDEX_ARRAY_POINTER = 0x8091;
    static constexpr uint32_t GL_INDEX_ARRAY_STRIDE = 0x8086;
    static constexpr uint32_t GL_INDEX_ARRAY_TYPE = 0x8085;
    static constexpr uint32_t GL_INDEX_BITS = 0x0D51;
    static constexpr uint32_t GL_INDEX_CLEAR_VALUE = 0x0C20;
    static constexpr uint32_t GL_INDEX_LOGIC_OP = 0x0BF1;
    static constexpr uint32_t GL_INDEX_MODE = 0x0C30;
    static constexpr uint32_t GL_INDEX_OFFSET = 0x0D13;
    static constexpr uint32_t GL_INDEX_SHIFT = 0x0D12;
    static constexpr uint32_t GL_INDEX_WRITEMASK = 0x0C21;
    static constexpr uint32_t GL_INFO_LOG_LENGTH = 0x8B84;
    static constexpr uint32_t GL_INNOCENT_CONTEXT_RESET = 0x8254;
    static constexpr uint32_t GL_INT = 0x1404;
    static constexpr uint32_t GL_INTENSITY = 0x8049;
    static constexpr uint32_t GL_INTENSITY12 = 0x804C;
    static constexpr uint32_t GL_INTENSITY16 = 0x804D;
    static constexpr uint32_t GL_INTENSITY4 = 0x804A;
    static constexpr uint32_t GL_INTENSITY8 = 0x804B;
    static constexpr uint32_t GL_INTERLEAVED_ATTRIBS = 0x8C8C;
    static constexpr uint32_t GL_INTERNALFORMAT_ALPHA_SIZE = 0x8274;
    static constexpr uint32_t GL_INTERNALFORMAT_ALPHA_TYPE = 0x827B;
    static constexpr uint32_t GL_INTERNALFORMAT_BLUE_SIZE = 0x8273;
    static constexpr uint32_t GL_INTERNALFORMAT_BLUE_TYPE = 0x827A;
    static constexpr uint32_t GL_INTERNALFORMAT_DEPTH_SIZE = 0x8275;
    static constexpr uint32_t GL_INTERNALFORMAT_DEPTH_TYPE = 0x827C;
    static constexpr uint32_t GL_INTERNALFORMAT_GREEN_SIZE = 0x8272;
    static constexpr uint32_t GL_INTERNALFORMAT_GREEN_TYPE = 0x8279;
    static constexpr uint32_t GL_INTERNALFORMAT_PREFERRED = 0x8270;
    static constexpr uint32_t GL_INTERNALFORMAT_RED_SIZE = 0x8271;
    static constexpr uint32_t GL_INTERNALFORMAT_RED_TYPE = 0x8278;
    static constexpr uint32_t GL_INTERNALFORMAT_SHARED_SIZE = 0x8277;
    static constexpr uint32_t GL_INTERNALFORMAT_STENCIL_SIZE = 0x8276;
    static constexpr uint32_t GL_INTERNALFORMAT_STENCIL_TYPE = 0x827D;
    static constexpr uint32_t GL_INTERNALFORMAT_SUPPORTED = 0x826F;
    static constexpr uint32_t GL_INTERPOLATE = 0x8575;
    static constexpr uint32_t GL_INT_2_10_10_10_REV = 0x8D9F;
    static constexpr uint32_t GL_INT_IMAGE_1D = 0x9057;
    static constexpr uint32_t GL_INT_IMAGE_1D_ARRAY = 0x905D;
    static constexpr uint32_t GL_INT_IMAGE_2D = 0x9058;
    static constexpr uint32_t GL_INT_IMAGE_2D_ARRAY = 0x905E;
    static constexpr uint32_t GL_INT_IMAGE_2D_MULTISAMPLE = 0x9060;
    static constexpr uint32_t GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY = 0x9061;
    static constexpr uint32_t GL_INT_IMAGE_2D_RECT = 0x905A;
    static constexpr uint32_t GL_INT_IMAGE_3D = 0x9059;
    static constexpr uint32_t GL_INT_IMAGE_BUFFER = 0x905C;
    static constexpr uint32_t GL_INT_IMAGE_CUBE = 0x905B;
    static constexpr uint32_t GL_INT_IMAGE_CUBE_MAP_ARRAY = 0x905F;
    static constexpr uint32_t GL_INT_SAMPLER_1D = 0x8DC9;
    static constexpr uint32_t GL_INT_SAMPLER_1D_ARRAY = 0x8DCE;
    static constexpr uint32_t GL_INT_SAMPLER_2D = 0x8DCA;
    static constexpr uint32_t GL_INT_SAMPLER_2D_ARRAY = 0x8DCF;
    static constexpr uint32_t GL_INT_SAMPLER_2D_MULTISAMPLE = 0x9109;
    static constexpr uint32_t GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY = 0x910C;
    static constexpr uint32_t GL_INT_SAMPLER_2D_RECT = 0x8DCD;
    static constexpr uint32_t GL_INT_SAMPLER_3D = 0x8DCB;
    static constexpr uint32_t GL_INT_SAMPLER_BUFFER = 0x8DD0;
    static constexpr uint32_t GL_INT_SAMPLER_CUBE = 0x8DCC;
    static constexpr uint32_t GL_INT_SAMPLER_CUBE_MAP_ARRAY = 0x900E;
    static constexpr uint32_t GL_INT_VEC2 = 0x8B53;
    static constexpr uint32_t GL_INT_VEC3 = 0x8B54;
    static constexpr uint32_t GL_INT_VEC4 = 0x8B55;
    static constexpr uint32_t GL_INVALID_ENUM = 0x0500;
    static constexpr uint32_t GL_INVALID_FRAMEBUFFER_OPERATION = 0x0506;
    static constexpr uint32_t GL_INVALID_INDEX = 0xFFFFFFFF;
    static constexpr uint32_t GL_INVALID_OPERATION = 0x0502;
    static constexpr uint32_t GL_INVALID_VALUE = 0x0501;
    static constexpr uint32_t GL_INVERT = 0x150A;
    static constexpr uint32_t GL_ISOLINES = 0x8E7A;
    static constexpr uint32_t GL_IS_PER_PATCH = 0x92E7;
    static constexpr uint32_t GL_IS_ROW_MAJOR = 0x9300;
    static constexpr uint32_t GL_KEEP = 0x1E00;
    static constexpr uint32_t GL_LAST_VERTEX_CONVENTION = 0x8E4E;
    static constexpr uint32_t GL_LAYER_PROVOKING_VERTEX = 0x825E;
    static constexpr uint32_t GL_LEFT = 0x0406;
    static constexpr uint32_t GL_LEQUAL = 0x0203;
    static constexpr uint32_t GL_LESS = 0x0201;
    static constexpr uint32_t GL_LIGHT0 = 0x4000;
    static constexpr uint32_t GL_LIGHT1 = 0x4001;
    static constexpr uint32_t GL_LIGHT2 = 0x4002;
    static constexpr uint32_t GL_LIGHT3 = 0x4003;
    static constexpr uint32_t GL_LIGHT4 = 0x4004;
    static constexpr uint32_t GL_LIGHT5 = 0x4005;
    static constexpr uint32_t GL_LIGHT6 = 0x4006;
    static constexpr uint32_t GL_LIGHT7 = 0x4007;
    static constexpr uint32_t GL_LIGHTING = 0x0B50;
    static constexpr uint32_t GL_LIGHTING_BIT = 0x00000040;
    static constexpr uint32_t GL_LIGHT_MODEL_AMBIENT = 0x0B53;
    static constexpr uint32_t GL_LIGHT_MODEL_COLOR_CONTROL = 0x81F8;
    static constexpr uint32_t GL_LIGHT_MODEL_LOCAL_VIEWER = 0x0B51;
    static constexpr uint32_t GL_LIGHT_MODEL_TWO_SIDE = 0x0B52;
    static constexpr uint32_t GL_LINE = 0x1B01;
    static constexpr uint32_t GL_LINEAR = 0x2601;
    static constexpr uint32_t GL_LINEAR_ATTENUATION = 0x1208;
    static constexpr uint32_t GL_LINEAR_MIPMAP_LINEAR = 0x2703;
    static constexpr uint32_t GL_LINEAR_MIPMAP_NEAREST = 0x2701;
    static constexpr uint32_t GL_LINES = 0x0001;
    static constexpr uint32_t GL_LINES_ADJACENCY = 0x000A;
    static constexpr uint32_t GL_LINE_BIT = 0x00000004;
    static constexpr uint32_t GL_LINE_LOOP = 0x0002;
    static constexpr uint32_t GL_LINE_RESET_TOKEN = 0x0707;
    static constexpr uint32_t GL_LINE_SMOOTH = 0x0B20;
    static constexpr uint32_t GL_LINE_SMOOTH_HINT = 0x0C52;
    static constexpr uint32_t GL_LINE_STIPPLE = 0x0B24;
    static constexpr uint32_t GL_LINE_STIPPLE_PATTERN = 0x0B25;
    static constexpr uint32_t GL_LINE_STIPPLE_REPEAT = 0x0B26;
    static constexpr uint32_t GL_LINE_STRIP = 0x0003;
    static constexpr uint32_t GL_LINE_STRIP_ADJACENCY = 0x000B;
    static constexpr uint32_t GL_LINE_TOKEN = 0x0702;
    static constexpr uint32_t GL_LINE_WIDTH = 0x0B21;
    static constexpr uint32_t GL_LINE_WIDTH_GRANULARITY = 0x0B23;
    static constexpr uint32_t GL_LINE_WIDTH_RANGE = 0x0B22;
    static constexpr uint32_t GL_LINK_STATUS = 0x8B82;
    static constexpr uint32_t GL_LIST_BASE = 0x0B32;
    static constexpr uint32_t GL_LIST_BIT = 0x00020000;
    static constexpr uint32_t GL_LIST_INDEX = 0x0B33;
    static constexpr uint32_t GL_LIST_MODE = 0x0B30;
    static constexpr uint32_t GL_LOAD = 0x0101;
    static constexpr uint32_t GL_LOCATION = 0x930E;
    static constexpr uint32_t GL_LOCATION_COMPONENT = 0x934A;
    static constexpr uint32_t GL_LOCATION_INDEX = 0x930F;
    static constexpr uint32_t GL_LOGIC_OP = 0x0BF1;
    static constexpr uint32_t GL_LOGIC_OP_MODE = 0x0BF0;
    static constexpr uint32_t GL_LOSE_CONTEXT_ON_RESET = 0x8252;
    static constexpr uint32_t GL_LOWER_LEFT = 0x8CA1;
    static constexpr uint32_t GL_LOW_FLOAT = 0x8DF0;
    static constexpr uint32_t GL_LOW_INT = 0x8DF3;
    static constexpr uint32_t GL_LUMINANCE = 0x1909;
    static constexpr uint32_t GL_LUMINANCE12 = 0x8041;
    static constexpr uint32_t GL_LUMINANCE12_ALPHA12 = 0x8047;
    static constexpr uint32_t GL_LUMINANCE12_ALPHA4 = 0x8046;
    static constexpr uint32_t GL_LUMINANCE16 = 0x8042;
    static constexpr uint32_t GL_LUMINANCE16_ALPHA16 = 0x8048;
    static constexpr uint32_t GL_LUMINANCE4 = 0x803F;
    static constexpr uint32_t GL_LUMINANCE4_ALPHA4 = 0x8043;
    static constexpr uint32_t GL_LUMINANCE6_ALPHA2 = 0x8044;
    static constexpr uint32_t GL_LUMINANCE8 = 0x8040;
    static constexpr uint32_t GL_LUMINANCE8_ALPHA8 = 0x8045;
    static constexpr uint32_t GL_LUMINANCE_ALPHA = 0x190A;
    static constexpr uint32_t GL_MAJOR_VERSION = 0x821B;
    static constexpr uint32_t GL_MANUAL_GENERATE_MIPMAP = 0x8294;
    static constexpr uint32_t GL_MAP1_COLOR_4 = 0x0D90;
    static constexpr uint32_t GL_MAP1_GRID_DOMAIN = 0x0DD0;
    static constexpr uint32_t GL_MAP1_GRID_SEGMENTS = 0x0DD1;
    static constexpr uint32_t GL_MAP1_INDEX = 0x0D91;
    static constexpr uint32_t GL_MAP1_NORMAL = 0x0D92;
    static constexpr uint32_t GL_MAP1_TEXTURE_COORD_1 = 0x0D93;
    static constexpr uint32_t GL_MAP1_TEXTURE_COORD_2 = 0x0D94;
    static constexpr uint32_t GL_MAP1_TEXTURE_COORD_3 = 0x0D95;
    static constexpr uint32_t GL_MAP1_TEXTURE_COORD_4 = 0x0D96;
    static constexpr uint32_t GL_MAP1_VERTEX_3 = 0x0D97;
    static constexpr uint32_t GL_MAP1_VERTEX_4 = 0x0D98;
    static constexpr uint32_t GL_MAP2_COLOR_4 = 0x0DB0;
    static constexpr uint32_t GL_MAP2_GRID_DOMAIN = 0x0DD2;
    static constexpr uint32_t GL_MAP2_GRID_SEGMENTS = 0x0DD3;
    static constexpr uint32_t GL_MAP2_INDEX = 0x0DB1;
    static constexpr uint32_t GL_MAP2_NORMAL = 0x0DB2;
    static constexpr uint32_t GL_MAP2_TEXTURE_COORD_1 = 0x0DB3;
    static constexpr uint32_t GL_MAP2_TEXTURE_COORD_2 = 0x0DB4;
    static constexpr uint32_t GL_MAP2_TEXTURE_COORD_3 = 0x0DB5;
    static constexpr uint32_t GL_MAP2_TEXTURE_COORD_4 = 0x0DB6;
    static constexpr uint32_t GL_MAP2_VERTEX_3 = 0x0DB7;
    static constexpr uint32_t GL_MAP2_VERTEX_4 = 0x0DB8;
    static constexpr uint32_t GL_MAP_COHERENT_BIT = 0x0080;
    static constexpr uint32_t GL_MAP_COLOR = 0x0D10;
    static constexpr uint32_t GL_MAP_FLUSH_EXPLICIT_BIT = 0x0010;
    static constexpr uint32_t GL_MAP_INVALIDATE_BUFFER_BIT = 0x0008;
    static constexpr uint32_t GL_MAP_INVALIDATE_RANGE_BIT = 0x0004;
    static constexpr uint32_t GL_MAP_PERSISTENT_BIT = 0x0040;
    static constexpr uint32_t GL_MAP_READ_BIT = 0x0001;
    static constexpr uint32_t GL_MAP_STENCIL = 0x0D11;
    static constexpr uint32_t GL_MAP_UNSYNCHRONIZED_BIT = 0x0020;
    static constexpr uint32_t GL_MAP_WRITE_BIT = 0x0002;
    static constexpr uint32_t GL_MATRIX_MODE = 0x0BA0;
    static constexpr uint32_t GL_MATRIX_STRIDE = 0x92FF;
    static constexpr uint32_t GL_MAX = 0x8008;
    static constexpr uint32_t GL_MAX_3D_TEXTURE_SIZE = 0x8073;
    static constexpr uint32_t GL_MAX_ARRAY_TEXTURE_LAYERS = 0x88FF;
    static constexpr uint32_t GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS = 0x92DC;
    static constexpr uint32_t GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE = 0x92D8;
    static constexpr uint32_t GL_MAX_ATTRIB_STACK_DEPTH = 0x0D35;
    static constexpr uint32_t GL_MAX_CLIENT_ATTRIB_STACK_DEPTH = 0x0D3B;
    static constexpr uint32_t GL_MAX_CLIP_DISTANCES = 0x0D32;
    static constexpr uint32_t GL_MAX_CLIP_PLANES = 0x0D32;
    static constexpr uint32_t GL_MAX_COLOR_ATTACHMENTS = 0x8CDF;
    static constexpr uint32_t GL_MAX_COLOR_TEXTURE_SAMPLES = 0x910E;
    static constexpr uint32_t GL_MAX_COMBINED_ATOMIC_COUNTERS = 0x92D7;
    static constexpr uint32_t GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS = 0x92D1;
    static constexpr uint32_t GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES = 0x82FA;
    static constexpr uint32_t GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS = 0x8266;
    static constexpr uint32_t GL_MAX_COMBINED_DIMENSIONS = 0x8282;
    static constexpr uint32_t GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS = 0x8A33;
    static constexpr uint32_t GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS = 0x8A32;
    static constexpr uint32_t GL_MAX_COMBINED_IMAGE_UNIFORMS = 0x90CF;
    static constexpr uint32_t GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS = 0x8F39;
    static constexpr uint32_t GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES = 0x8F39;
    static constexpr uint32_t GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS = 0x90DC;
    static constexpr uint32_t GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS = 0x8E1E;
    static constexpr uint32_t GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS = 0x8E1F;
    static constexpr uint32_t GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS = 0x8B4D;
    static constexpr uint32_t GL_MAX_COMBINED_UNIFORM_BLOCKS = 0x8A2E;
    static constexpr uint32_t GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS = 0x8A31;
    static constexpr uint32_t GL_MAX_COMPUTE_ATOMIC_COUNTERS = 0x8265;
    static constexpr uint32_t GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS = 0x8264;
    static constexpr uint32_t GL_MAX_COMPUTE_IMAGE_UNIFORMS = 0x91BD;
    static constexpr uint32_t GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS = 0x90DB;
    static constexpr uint32_t GL_MAX_COMPUTE_SHARED_MEMORY_SIZE = 0x8262;
    static constexpr uint32_t GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS = 0x91BC;
    static constexpr uint32_t GL_MAX_COMPUTE_UNIFORM_BLOCKS = 0x91BB;
    static constexpr uint32_t GL_MAX_COMPUTE_UNIFORM_COMPONENTS = 0x8263;
    static constexpr uint32_t GL_MAX_COMPUTE_WORK_GROUP_COUNT = 0x91BE;
    static constexpr uint32_t GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS = 0x90EB;
    static constexpr uint32_t GL_MAX_COMPUTE_WORK_GROUP_SIZE = 0x91BF;
    static constexpr uint32_t GL_MAX_CUBE_MAP_TEXTURE_SIZE = 0x851C;
    static constexpr uint32_t GL_MAX_CULL_DISTANCES = 0x82F9;
    static constexpr uint32_t GL_MAX_DEBUG_GROUP_STACK_DEPTH = 0x826C;
    static constexpr uint32_t GL_MAX_DEBUG_LOGGED_MESSAGES = 0x9144;
    static constexpr uint32_t GL_MAX_DEBUG_MESSAGE_LENGTH = 0x9143;
    static constexpr uint32_t GL_MAX_DEPTH = 0x8280;
    static constexpr uint32_t GL_MAX_DEPTH_TEXTURE_SAMPLES = 0x910F;
    static constexpr uint32_t GL_MAX_DRAW_BUFFERS = 0x8824;
    static constexpr uint32_t GL_MAX_DUAL_SOURCE_DRAW_BUFFERS = 0x88FC;
    static constexpr uint32_t GL_MAX_ELEMENTS_INDICES = 0x80E9;
    static constexpr uint32_t GL_MAX_ELEMENTS_VERTICES = 0x80E8;
    static constexpr uint32_t GL_MAX_ELEMENT_INDEX = 0x8D6B;
    static constexpr uint32_t GL_MAX_EVAL_ORDER = 0x0D30;
    static constexpr uint32_t GL_MAX_FRAGMENT_ATOMIC_COUNTERS = 0x92D6;
    static constexpr uint32_t GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS = 0x92D0;
    static constexpr uint32_t GL_MAX_FRAGMENT_IMAGE_UNIFORMS = 0x90CE;
    static constexpr uint32_t GL_MAX_FRAGMENT_INPUT_COMPONENTS = 0x9125;
    static constexpr uint32_t GL_MAX_FRAGMENT_INTERPOLATION_OFFSET = 0x8E5C;
    static constexpr uint32_t GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS = 0x90DA;
    static constexpr uint32_t GL_MAX_FRAGMENT_UNIFORM_BLOCKS = 0x8A2D;
    static constexpr uint32_t GL_MAX_FRAGMENT_UNIFORM_COMPONENTS = 0x8B49;
    static constexpr uint32_t GL_MAX_FRAGMENT_UNIFORM_VECTORS = 0x8DFD;
    static constexpr uint32_t GL_MAX_FRAMEBUFFER_HEIGHT = 0x9316;
    static constexpr uint32_t GL_MAX_FRAMEBUFFER_LAYERS = 0x9317;
    static constexpr uint32_t GL_MAX_FRAMEBUFFER_SAMPLES = 0x9318;
    static constexpr uint32_t GL_MAX_FRAMEBUFFER_WIDTH = 0x9315;
    static constexpr uint32_t GL_MAX_GEOMETRY_ATOMIC_COUNTERS = 0x92D5;
    static constexpr uint32_t GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS = 0x92CF;
    static constexpr uint32_t GL_MAX_GEOMETRY_IMAGE_UNIFORMS = 0x90CD;
    static constexpr uint32_t GL_MAX_GEOMETRY_INPUT_COMPONENTS = 0x9123;
    static constexpr uint32_t GL_MAX_GEOMETRY_OUTPUT_COMPONENTS = 0x9124;
    static constexpr uint32_t GL_MAX_GEOMETRY_OUTPUT_VERTICES = 0x8DE0;
    static constexpr uint32_t GL_MAX_GEOMETRY_SHADER_INVOCATIONS = 0x8E5A;
    static constexpr uint32_t GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS = 0x90D7;
    static constexpr uint32_t GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS = 0x8C29;
    static constexpr uint32_t GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS = 0x8DE1;
    static constexpr uint32_t GL_MAX_GEOMETRY_UNIFORM_BLOCKS = 0x8A2C;
    static constexpr uint32_t GL_MAX_GEOMETRY_UNIFORM_COMPONENTS = 0x8DDF;
    static constexpr uint32_t GL_MAX_HEIGHT = 0x827F;
    static constexpr uint32_t GL_MAX_IMAGE_SAMPLES = 0x906D;
    static constexpr uint32_t GL_MAX_IMAGE_UNITS = 0x8F38;
    static constexpr uint32_t GL_MAX_INTEGER_SAMPLES = 0x9110;
    static constexpr uint32_t GL_MAX_LABEL_LENGTH = 0x82E8;
    static constexpr uint32_t GL_MAX_LAYERS = 0x8281;
    static constexpr uint32_t GL_MAX_LIGHTS = 0x0D31;
    static constexpr uint32_t GL_MAX_LIST_NESTING = 0x0B31;
    static constexpr uint32_t GL_MAX_MODELVIEW_STACK_DEPTH = 0x0D36;
    static constexpr uint32_t GL_MAX_NAME_LENGTH = 0x92F6;
    static constexpr uint32_t GL_MAX_NAME_STACK_DEPTH = 0x0D37;
    static constexpr uint32_t GL_MAX_NUM_ACTIVE_VARIABLES = 0x92F7;
    static constexpr uint32_t GL_MAX_NUM_COMPATIBLE_SUBROUTINES = 0x92F8;
    static constexpr uint32_t GL_MAX_PATCH_VERTICES = 0x8E7D;
    static constexpr uint32_t GL_MAX_PIXEL_MAP_TABLE = 0x0D34;
    static constexpr uint32_t GL_MAX_PROGRAM_TEXEL_OFFSET = 0x8905;
    static constexpr uint32_t GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET = 0x8E5F;
    static constexpr uint32_t GL_MAX_PROJECTION_STACK_DEPTH = 0x0D38;
    static constexpr uint32_t GL_MAX_RECTANGLE_TEXTURE_SIZE = 0x84F8;
    static constexpr uint32_t GL_MAX_RENDERBUFFER_SIZE = 0x84E8;
    static constexpr uint32_t GL_MAX_SAMPLES = 0x8D57;
    static constexpr uint32_t GL_MAX_SAMPLE_MASK_WORDS = 0x8E59;
    static constexpr uint32_t GL_MAX_SERVER_WAIT_TIMEOUT = 0x9111;
    static constexpr uint32_t GL_MAX_SHADER_STORAGE_BLOCK_SIZE = 0x90DE;
    static constexpr uint32_t GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS = 0x90DD;
    static constexpr uint32_t GL_MAX_SUBROUTINES = 0x8DE7;
    static constexpr uint32_t GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS = 0x8DE8;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS = 0x92D3;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS = 0x92CD;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS = 0x90CB;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_INPUT_COMPONENTS = 0x886C;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS = 0x8E83;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS = 0x90D8;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS = 0x8E81;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS = 0x8E85;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS = 0x8E89;
    static constexpr uint32_t GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS = 0x8E7F;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS = 0x92D4;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS = 0x92CE;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS = 0x90CC;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS = 0x886D;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS = 0x8E86;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS = 0x90D9;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS = 0x8E82;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS = 0x8E8A;
    static constexpr uint32_t GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS = 0x8E80;
    static constexpr uint32_t GL_MAX_TESS_GEN_LEVEL = 0x8E7E;
    static constexpr uint32_t GL_MAX_TESS_PATCH_COMPONENTS = 0x8E84;
    static constexpr uint32_t GL_MAX_TEXTURE_BUFFER_SIZE = 0x8C2B;
    static constexpr uint32_t GL_MAX_TEXTURE_COORDS = 0x8871;
    static constexpr uint32_t GL_MAX_TEXTURE_IMAGE_UNITS = 0x8872;
    static constexpr uint32_t GL_MAX_TEXTURE_LOD_BIAS = 0x84FD;
    static constexpr uint32_t GL_MAX_TEXTURE_MAX_ANISOTROPY = 0x84FF;
    static constexpr uint32_t GL_MAX_TEXTURE_SIZE = 0x0D33;
    static constexpr uint32_t GL_MAX_TEXTURE_STACK_DEPTH = 0x0D39;
    static constexpr uint32_t GL_MAX_TEXTURE_UNITS = 0x84E2;
    static constexpr uint32_t GL_MAX_TRANSFORM_FEEDBACK_BUFFERS = 0x8E70;
    static constexpr uint32_t GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS = 0x8C8A;
    static constexpr uint32_t GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS = 0x8C8B;
    static constexpr uint32_t GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS = 0x8C80;
    static constexpr uint32_t GL_MAX_UNIFORM_BLOCK_SIZE = 0x8A30;
    static constexpr uint32_t GL_MAX_UNIFORM_BUFFER_BINDINGS = 0x8A2F;
    static constexpr uint32_t GL_MAX_UNIFORM_LOCATIONS = 0x826E;
    static constexpr uint32_t GL_MAX_VARYING_COMPONENTS = 0x8B4B;
    static constexpr uint32_t GL_MAX_VARYING_FLOATS = 0x8B4B;
    static constexpr uint32_t GL_MAX_VARYING_VECTORS = 0x8DFC;
    static constexpr uint32_t GL_MAX_VERTEX_ATOMIC_COUNTERS = 0x92D2;
    static constexpr uint32_t GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS = 0x92CC;
    static constexpr uint32_t GL_MAX_VERTEX_ATTRIBS = 0x8869;
    static constexpr uint32_t GL_MAX_VERTEX_ATTRIB_BINDINGS = 0x82DA;
    static constexpr uint32_t GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET = 0x82D9;
    static constexpr uint32_t GL_MAX_VERTEX_ATTRIB_STRIDE = 0x82E5;
    static constexpr uint32_t GL_MAX_VERTEX_IMAGE_UNIFORMS = 0x90CA;
    static constexpr uint32_t GL_MAX_VERTEX_OUTPUT_COMPONENTS = 0x9122;
    static constexpr uint32_t GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS = 0x90D6;
    static constexpr uint32_t GL_MAX_VERTEX_STREAMS = 0x8E71;
    static constexpr uint32_t GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS = 0x8B4C;
    static constexpr uint32_t GL_MAX_VERTEX_UNIFORM_BLOCKS = 0x8A2B;
    static constexpr uint32_t GL_MAX_VERTEX_UNIFORM_COMPONENTS = 0x8B4A;
    static constexpr uint32_t GL_MAX_VERTEX_UNIFORM_VECTORS = 0x8DFB;
    static constexpr uint32_t GL_MAX_VIEWPORTS = 0x825B;
    static constexpr uint32_t GL_MAX_VIEWPORT_DIMS = 0x0D3A;
    static constexpr uint32_t GL_MAX_WIDTH = 0x827E;
    static constexpr uint32_t GL_MEDIUM_FLOAT = 0x8DF1;
    static constexpr uint32_t GL_MEDIUM_INT = 0x8DF4;
    static constexpr uint32_t GL_MIN = 0x8007;
    static constexpr uint32_t GL_MINMAX = 0x802E;
    static constexpr uint32_t GL_MINOR_VERSION = 0x821C;
    static constexpr uint32_t GL_MIN_FRAGMENT_INTERPOLATION_OFFSET = 0x8E5B;
    static constexpr uint32_t GL_MIN_MAP_BUFFER_ALIGNMENT = 0x90BC;
    static constexpr uint32_t GL_MIN_PROGRAM_TEXEL_OFFSET = 0x8904;
    static constexpr uint32_t GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET = 0x8E5E;
    static constexpr uint32_t GL_MIN_SAMPLE_SHADING_VALUE = 0x8C37;
    static constexpr uint32_t GL_MIPMAP = 0x8293;
    static constexpr uint32_t GL_MIRRORED_REPEAT = 0x8370;
    static constexpr uint32_t GL_MIRROR_CLAMP_TO_EDGE = 0x8743;
    static constexpr uint32_t GL_MODELVIEW = 0x1700;
    static constexpr uint32_t GL_MODELVIEW_MATRIX = 0x0BA6;
    static constexpr uint32_t GL_MODELVIEW_STACK_DEPTH = 0x0BA3;
    static constexpr uint32_t GL_MODULATE = 0x2100;
    static constexpr uint32_t GL_MULT = 0x0103;
    static constexpr uint32_t GL_MULTISAMPLE = 0x809D;
    static constexpr uint32_t GL_MULTISAMPLE_BIT = 0x20000000;
    static constexpr uint32_t GL_N3F_V3F = 0x2A25;
    static constexpr uint32_t GL_NAME_LENGTH = 0x92F9;
    static constexpr uint32_t GL_NAME_STACK_DEPTH = 0x0D70;
    static constexpr uint32_t GL_NAND = 0x150E;
    static constexpr uint32_t GL_NEAREST = 0x2600;
    static constexpr uint32_t GL_NEAREST_MIPMAP_LINEAR = 0x2702;
    static constexpr uint32_t GL_NEAREST_MIPMAP_NEAREST = 0x2700;
    static constexpr uint32_t GL_NEGATIVE_ONE_TO_ONE = 0x935E;
    static constexpr uint32_t GL_NEVER = 0x0200;
    static constexpr uint32_t GL_NICEST = 0x1102;
    static constexpr uint32_t GL_NONE = 0;
    static constexpr uint32_t GL_NOOP = 0x1505;
    static constexpr uint32_t GL_NOR = 0x1508;
    static constexpr uint32_t GL_NORMALIZE = 0x0BA1;
    static constexpr uint32_t GL_NORMAL_ARRAY = 0x8075;
    static constexpr uint32_t GL_NORMAL_ARRAY_BUFFER_BINDING = 0x8897;
    static constexpr uint32_t GL_NORMAL_ARRAY_POINTER = 0x808F;
    static constexpr uint32_t GL_NORMAL_ARRAY_STRIDE = 0x807F;
    static constexpr uint32_t GL_NORMAL_ARRAY_TYPE = 0x807E;
    static constexpr uint32_t GL_NORMAL_MAP = 0x8511;
    static constexpr uint32_t GL_NOTEQUAL = 0x0205;
    static constexpr uint32_t GL_NO_ERROR = 0;
    static constexpr uint32_t GL_NO_RESET_NOTIFICATION = 0x8261;
    static constexpr uint32_t GL_NUM_ACTIVE_VARIABLES = 0x9304;
    static constexpr uint32_t GL_NUM_COMPATIBLE_SUBROUTINES = 0x8E4A;
    static constexpr uint32_t GL_NUM_COMPRESSED_TEXTURE_FORMATS = 0x86A2;
    static constexpr uint32_t GL_NUM_EXTENSIONS = 0x821D;
    static constexpr uint32_t GL_NUM_PROGRAM_BINARY_FORMATS = 0x87FE;
    static constexpr uint32_t GL_NUM_SAMPLE_COUNTS = 0x9380;
    static constexpr uint32_t GL_NUM_SHADER_BINARY_FORMATS = 0x8DF9;
    static constexpr uint32_t GL_NUM_SHADING_LANGUAGE_VERSIONS = 0x82E9;
    static constexpr uint32_t GL_NUM_SPIR_V_EXTENSIONS = 0x9554;
    static constexpr uint32_t GL_OBJECT_LINEAR = 0x2401;
    static constexpr uint32_t GL_OBJECT_PLANE = 0x2501;
    static constexpr uint32_t GL_OBJECT_TYPE = 0x9112;
    static constexpr uint32_t GL_OFFSET = 0x92FC;
    static constexpr uint32_t GL_ONE = 1;
    static constexpr uint32_t GL_ONE_MINUS_CONSTANT_ALPHA = 0x8004;
    static constexpr uint32_t GL_ONE_MINUS_CONSTANT_COLOR = 0x8002;
    static constexpr uint32_t GL_ONE_MINUS_DST_ALPHA = 0x0305;
    static constexpr uint32_t GL_ONE_MINUS_DST_COLOR = 0x0307;
    static constexpr uint32_t GL_ONE_MINUS_SRC1_ALPHA = 0x88FB;
    static constexpr uint32_t GL_ONE_MINUS_SRC1_COLOR = 0x88FA;
    static constexpr uint32_t GL_ONE_MINUS_SRC_ALPHA = 0x0303;
    static constexpr uint32_t GL_ONE_MINUS_SRC_COLOR = 0x0301;
    static constexpr uint32_t GL_OPERAND0_ALPHA = 0x8598;
    static constexpr uint32_t GL_OPERAND0_RGB = 0x8590;
    static constexpr uint32_t GL_OPERAND1_ALPHA = 0x8599;
    static constexpr uint32_t GL_OPERAND1_RGB = 0x8591;
    static constexpr uint32_t GL_OPERAND2_ALPHA = 0x859A;
    static constexpr uint32_t GL_OPERAND2_RGB = 0x8592;
    static constexpr uint32_t GL_OR = 0x1507;
    static constexpr uint32_t GL_ORDER = 0x0A01;
    static constexpr uint32_t GL_OR_INVERTED = 0x150D;
    static constexpr uint32_t GL_OR_REVERSE = 0x150B;
    static constexpr uint32_t GL_OUT_OF_MEMORY = 0x0505;
    static constexpr uint32_t GL_PACK_ALIGNMENT = 0x0D05;
    static constexpr uint32_t GL_PACK_COMPRESSED_BLOCK_DEPTH = 0x912D;
    static constexpr uint32_t GL_PACK_COMPRESSED_BLOCK_HEIGHT = 0x912C;
    static constexpr uint32_t GL_PACK_COMPRESSED_BLOCK_SIZE = 0x912E;
    static constexpr uint32_t GL_PACK_COMPRESSED_BLOCK_WIDTH = 0x912B;
    static constexpr uint32_t GL_PACK_IMAGE_HEIGHT = 0x806C;
    static constexpr uint32_t GL_PACK_LSB_FIRST = 0x0D01;
    static constexpr uint32_t GL_PACK_ROW_LENGTH = 0x0D02;
    static constexpr uint32_t GL_PACK_SKIP_IMAGES = 0x806B;
    static constexpr uint32_t GL_PACK_SKIP_PIXELS = 0x0D04;
    static constexpr uint32_t GL_PACK_SKIP_ROWS = 0x0D03;
    static constexpr uint32_t GL_PACK_SWAP_BYTES = 0x0D00;
    static constexpr uint32_t GL_PARAMETER_BUFFER = 0x80EE;
    static constexpr uint32_t GL_PARAMETER_BUFFER_BINDING = 0x80EF;
    static constexpr uint32_t GL_PASS_THROUGH_TOKEN = 0x0700;
    static constexpr uint32_t GL_PATCHES = 0x000E;
    static constexpr uint32_t GL_PATCH_DEFAULT_INNER_LEVEL = 0x8E73;
    static constexpr uint32_t GL_PATCH_DEFAULT_OUTER_LEVEL = 0x8E74;
    static constexpr uint32_t GL_PATCH_VERTICES = 0x8E72;
    static constexpr uint32_t GL_PERSPECTIVE_CORRECTION_HINT = 0x0C50;
    static constexpr uint32_t GL_PIXEL_BUFFER_BARRIER_BIT = 0x00000080;
    static constexpr uint32_t GL_PIXEL_MAP_A_TO_A = 0x0C79;
    static constexpr uint32_t GL_PIXEL_MAP_A_TO_A_SIZE = 0x0CB9;
    static constexpr uint32_t GL_PIXEL_MAP_B_TO_B = 0x0C78;
    static constexpr uint32_t GL_PIXEL_MAP_B_TO_B_SIZE = 0x0CB8;
    static constexpr uint32_t GL_PIXEL_MAP_G_TO_G = 0x0C77;
    static constexpr uint32_t GL_PIXEL_MAP_G_TO_G_SIZE = 0x0CB7;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_A = 0x0C75;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_A_SIZE = 0x0CB5;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_B = 0x0C74;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_B_SIZE = 0x0CB4;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_G = 0x0C73;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_G_SIZE = 0x0CB3;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_I = 0x0C70;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_I_SIZE = 0x0CB0;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_R = 0x0C72;
    static constexpr uint32_t GL_PIXEL_MAP_I_TO_R_SIZE = 0x0CB2;
    static constexpr uint32_t GL_PIXEL_MAP_R_TO_R = 0x0C76;
    static constexpr uint32_t GL_PIXEL_MAP_R_TO_R_SIZE = 0x0CB6;
    static constexpr uint32_t GL_PIXEL_MAP_S_TO_S = 0x0C71;
    static constexpr uint32_t GL_PIXEL_MAP_S_TO_S_SIZE = 0x0CB1;
    static constexpr uint32_t GL_PIXEL_MODE_BIT = 0x00000020;
    static constexpr uint32_t GL_PIXEL_PACK_BUFFER = 0x88EB;
    static constexpr uint32_t GL_PIXEL_PACK_BUFFER_BINDING = 0x88ED;
    static constexpr uint32_t GL_PIXEL_UNPACK_BUFFER = 0x88EC;
    static constexpr uint32_t GL_PIXEL_UNPACK_BUFFER_BINDING = 0x88EF;
    static constexpr uint32_t GL_POINT = 0x1B00;
    static constexpr uint32_t GL_POINTS = 0x0000;
    static constexpr uint32_t GL_POINT_BIT = 0x00000002;
    static constexpr uint32_t GL_POINT_DISTANCE_ATTENUATION = 0x8129;
    static constexpr uint32_t GL_POINT_FADE_THRESHOLD_SIZE = 0x8128;
    static constexpr uint32_t GL_POINT_SIZE = 0x0B11;
    static constexpr uint32_t GL_POINT_SIZE_GRANULARITY = 0x0B13;
    static constexpr uint32_t GL_POINT_SIZE_MAX = 0x8127;
    static constexpr uint32_t GL_POINT_SIZE_MIN = 0x8126;
    static constexpr uint32_t GL_POINT_SIZE_RANGE = 0x0B12;
    static constexpr uint32_t GL_POINT_SMOOTH = 0x0B10;
    static constexpr uint32_t GL_POINT_SMOOTH_HINT = 0x0C51;
    static constexpr uint32_t GL_POINT_SPRITE = 0x8861;
    static constexpr uint32_t GL_POINT_SPRITE_COORD_ORIGIN = 0x8CA0;
    static constexpr uint32_t GL_POINT_TOKEN = 0x0701;
    static constexpr uint32_t GL_POLYGON = 0x0009;
    static constexpr uint32_t GL_POLYGON_BIT = 0x00000008;
    static constexpr uint32_t GL_POLYGON_MODE = 0x0B40;
    static constexpr uint32_t GL_POLYGON_OFFSET_CLAMP = 0x8E1B;
    static constexpr uint32_t GL_POLYGON_OFFSET_FACTOR = 0x8038;
    static constexpr uint32_t GL_POLYGON_OFFSET_FILL = 0x8037;
    static constexpr uint32_t GL_POLYGON_OFFSET_LINE = 0x2A02;
    static constexpr uint32_t GL_POLYGON_OFFSET_POINT = 0x2A01;
    static constexpr uint32_t GL_POLYGON_OFFSET_UNITS = 0x2A00;
    static constexpr uint32_t GL_POLYGON_SMOOTH = 0x0B41;
    static constexpr uint32_t GL_POLYGON_SMOOTH_HINT = 0x0C53;
    static constexpr uint32_t GL_POLYGON_STIPPLE = 0x0B42;
    static constexpr uint32_t GL_POLYGON_STIPPLE_BIT = 0x00000010;
    static constexpr uint32_t GL_POLYGON_TOKEN = 0x0703;
    static constexpr uint32_t GL_POSITION = 0x1203;
    static constexpr uint32_t GL_POST_COLOR_MATRIX_COLOR_TABLE = 0x80D2;
    static constexpr uint32_t GL_POST_CONVOLUTION_COLOR_TABLE = 0x80D1;
    static constexpr uint32_t GL_PREVIOUS = 0x8578;
    static constexpr uint32_t GL_PRIMARY_COLOR = 0x8577;
    static constexpr uint32_t GL_PRIMITIVES_GENERATED = 0x8C87;
    static constexpr uint32_t GL_PRIMITIVES_SUBMITTED = 0x82EF;
    static constexpr uint32_t GL_PRIMITIVE_RESTART = 0x8F9D;
    static constexpr uint32_t GL_PRIMITIVE_RESTART_FIXED_INDEX = 0x8D69;
    static constexpr uint32_t GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED = 0x8221;
    static constexpr uint32_t GL_PRIMITIVE_RESTART_INDEX = 0x8F9E;
    static constexpr uint32_t GL_PROGRAM = 0x82E2;
    static constexpr uint32_t GL_PROGRAM_BINARY_FORMATS = 0x87FF;
    static constexpr uint32_t GL_PROGRAM_BINARY_LENGTH = 0x8741;
    static constexpr uint32_t GL_PROGRAM_BINARY_RETRIEVABLE_HINT = 0x8257;
    static constexpr uint32_t GL_PROGRAM_INPUT = 0x92E3;
    static constexpr uint32_t GL_PROGRAM_OUTPUT = 0x92E4;
    static constexpr uint32_t GL_PROGRAM_PIPELINE = 0x82E4;
    static constexpr uint32_t GL_PROGRAM_PIPELINE_BINDING = 0x825A;
    static constexpr uint32_t GL_PROGRAM_POINT_SIZE = 0x8642;
    static constexpr uint32_t GL_PROGRAM_SEPARABLE = 0x8258;
    static constexpr uint32_t GL_PROJECTION = 0x1701;
    static constexpr uint32_t GL_PROJECTION_MATRIX = 0x0BA7;
    static constexpr uint32_t GL_PROJECTION_STACK_DEPTH = 0x0BA4;
    static constexpr uint32_t GL_PROVOKING_VERTEX = 0x8E4F;
    static constexpr uint32_t GL_PROXY_COLOR_TABLE = 0x80D3;
    static constexpr uint32_t GL_PROXY_HISTOGRAM = 0x8025;
    static constexpr uint32_t GL_PROXY_POST_COLOR_MATRIX_COLOR_TABLE = 0x80D5;
    static constexpr uint32_t GL_PROXY_POST_CONVOLUTION_COLOR_TABLE = 0x80D4;
    static constexpr uint32_t GL_PROXY_TEXTURE_1D = 0x8063;
    static constexpr uint32_t GL_PROXY_TEXTURE_1D_ARRAY = 0x8C19;
    static constexpr uint32_t GL_PROXY_TEXTURE_2D = 0x8064;
    static constexpr uint32_t GL_PROXY_TEXTURE_2D_ARRAY = 0x8C1B;
    static constexpr uint32_t GL_PROXY_TEXTURE_2D_MULTISAMPLE = 0x9101;
    static constexpr uint32_t GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY = 0x9103;
    static constexpr uint32_t GL_PROXY_TEXTURE_3D = 0x8070;
    static constexpr uint32_t GL_PROXY_TEXTURE_CUBE_MAP = 0x851B;
    static constexpr uint32_t GL_PROXY_TEXTURE_CUBE_MAP_ARRAY = 0x900B;
    static constexpr uint32_t GL_PROXY_TEXTURE_RECTANGLE = 0x84F7;
    static constexpr uint32_t GL_Q = 0x2003;
    static constexpr uint32_t GL_QUADRATIC_ATTENUATION = 0x1209;
    static constexpr uint32_t GL_QUADS = 0x0007;
    static constexpr uint32_t GL_QUADS_FOLLOW_PROVOKING_VERTEX_CONVENTION = 0x8E4C;
    static constexpr uint32_t GL_QUAD_STRIP = 0x0008;
    static constexpr uint32_t GL_QUERY = 0x82E3;
    static constexpr uint32_t GL_QUERY_BUFFER = 0x9192;
    static constexpr uint32_t GL_QUERY_BUFFER_BARRIER_BIT = 0x00008000;
    static constexpr uint32_t GL_QUERY_BUFFER_BINDING = 0x9193;
    static constexpr uint32_t GL_QUERY_BY_REGION_NO_WAIT = 0x8E16;
    static constexpr uint32_t GL_QUERY_BY_REGION_NO_WAIT_INVERTED = 0x8E1A;
    static constexpr uint32_t GL_QUERY_BY_REGION_WAIT = 0x8E15;
    static constexpr uint32_t GL_QUERY_BY_REGION_WAIT_INVERTED = 0x8E19;
    static constexpr uint32_t GL_QUERY_COUNTER_BITS = 0x8864;
    static constexpr uint32_t GL_QUERY_NO_WAIT = 0x8E14;
    static constexpr uint32_t GL_QUERY_NO_WAIT_INVERTED = 0x8E18;
    static constexpr uint32_t GL_QUERY_RESULT = 0x8866;
    static constexpr uint32_t GL_QUERY_RESULT_AVAILABLE = 0x8867;
    static constexpr uint32_t GL_QUERY_RESULT_NO_WAIT = 0x9194;
    static constexpr uint32_t GL_QUERY_TARGET = 0x82EA;
    static constexpr uint32_t GL_QUERY_WAIT = 0x8E13;
    static constexpr uint32_t GL_QUERY_WAIT_INVERTED = 0x8E17;
    static constexpr uint32_t GL_R = 0x2002;
    static constexpr uint32_t GL_R11F_G11F_B10F = 0x8C3A;
    static constexpr uint32_t GL_R16 = 0x822A;
    static constexpr uint32_t GL_R16F = 0x822D;
    static constexpr uint32_t GL_R16I = 0x8233;
    static constexpr uint32_t GL_R16UI = 0x8234;
    static constexpr uint32_t GL_R16_SNORM = 0x8F98;
    static constexpr uint32_t GL_R32F = 0x822E;
    static constexpr uint32_t GL_R32I = 0x8235;
    static constexpr uint32_t GL_R32UI = 0x8236;
    static constexpr uint32_t GL_R3_G3_B2 = 0x2A10;
    static constexpr uint32_t GL_R8 = 0x8229;
    static constexpr uint32_t GL_R8I = 0x8231;
    static constexpr uint32_t GL_R8UI = 0x8232;
    static constexpr uint32_t GL_R8_SNORM = 0x8F94;
    static constexpr uint32_t GL_RASTERIZER_DISCARD = 0x8C89;
    static constexpr uint32_t GL_READ_BUFFER = 0x0C02;
    static constexpr uint32_t GL_READ_FRAMEBUFFER = 0x8CA8;
    static constexpr uint32_t GL_READ_FRAMEBUFFER_BINDING = 0x8CAA;
    static constexpr uint32_t GL_READ_ONLY = 0x88B8;
    static constexpr uint32_t GL_READ_PIXELS = 0x828C;
    static constexpr uint32_t GL_READ_PIXELS_FORMAT = 0x828D;
    static constexpr uint32_t GL_READ_PIXELS_TYPE = 0x828E;
    static constexpr uint32_t GL_READ_WRITE = 0x88BA;
    static constexpr uint32_t GL_RED = 0x1903;
    static constexpr uint32_t GL_RED_BIAS = 0x0D15;
    static constexpr uint32_t GL_RED_BITS = 0x0D52;
    static constexpr uint32_t GL_RED_INTEGER = 0x8D94;
    static constexpr uint32_t GL_RED_SCALE = 0x0D14;
    static constexpr uint32_t GL_REFERENCED_BY_COMPUTE_SHADER = 0x930B;
    static constexpr uint32_t GL_REFERENCED_BY_FRAGMENT_SHADER = 0x930A;
    static constexpr uint32_t GL_REFERENCED_BY_GEOMETRY_SHADER = 0x9309;
    static constexpr uint32_t GL_REFERENCED_BY_TESS_CONTROL_SHADER = 0x9307;
    static constexpr uint32_t GL_REFERENCED_BY_TESS_EVALUATION_SHADER = 0x9308;
    static constexpr uint32_t GL_REFERENCED_BY_VERTEX_SHADER = 0x9306;
    static constexpr uint32_t GL_REFLECTION_MAP = 0x8512;
    static constexpr uint32_t GL_RENDER = 0x1C00;
    static constexpr uint32_t GL_RENDERBUFFER = 0x8D41;
    static constexpr uint32_t GL_RENDERBUFFER_ALPHA_SIZE = 0x8D53;
    static constexpr uint32_t GL_RENDERBUFFER_BINDING = 0x8CA7;
    static constexpr uint32_t GL_RENDERBUFFER_BLUE_SIZE = 0x8D52;
    static constexpr uint32_t GL_RENDERBUFFER_DEPTH_SIZE = 0x8D54;
    static constexpr uint32_t GL_RENDERBUFFER_GREEN_SIZE = 0x8D51;
    static constexpr uint32_t GL_RENDERBUFFER_HEIGHT = 0x8D43;
    static constexpr uint32_t GL_RENDERBUFFER_INTERNAL_FORMAT = 0x8D44;
    static constexpr uint32_t GL_RENDERBUFFER_RED_SIZE = 0x8D50;
    static constexpr uint32_t GL_RENDERBUFFER_SAMPLES = 0x8CAB;
    static constexpr uint32_t GL_RENDERBUFFER_STENCIL_SIZE = 0x8D55;
    static constexpr uint32_t GL_RENDERBUFFER_WIDTH = 0x8D42;
    static constexpr uint32_t GL_RENDERER = 0x1F01;
    static constexpr uint32_t GL_RENDER_MODE = 0x0C40;
    static constexpr uint32_t GL_REPEAT = 0x2901;
    static constexpr uint32_t GL_REPLACE = 0x1E01;
    static constexpr uint32_t GL_RESCALE_NORMAL = 0x803A;
    static constexpr uint32_t GL_RESET_NOTIFICATION_STRATEGY = 0x8256;
    static constexpr uint32_t GL_RETURN = 0x0102;
    static constexpr uint32_t GL_RG = 0x8227;
    static constexpr uint32_t GL_RG16 = 0x822C;
    static constexpr uint32_t GL_RG16F = 0x822F;
    static constexpr uint32_t GL_RG16I = 0x8239;
    static constexpr uint32_t GL_RG16UI = 0x823A;
    static constexpr uint32_t GL_RG16_SNORM = 0x8F99;
    static constexpr uint32_t GL_RG32F = 0x8230;
    static constexpr uint32_t GL_RG32I = 0x823B;
    static constexpr uint32_t GL_RG32UI = 0x823C;
    static constexpr uint32_t GL_RG8 = 0x822B;
    static constexpr uint32_t GL_RG8I = 0x8237;
    static constexpr uint32_t GL_RG8UI = 0x8238;
    static constexpr uint32_t GL_RG8_SNORM = 0x8F95;
    static constexpr uint32_t GL_RGB = 0x1907;
    static constexpr uint32_t GL_RGB10 = 0x8052;
    static constexpr uint32_t GL_RGB10_A2 = 0x8059;
    static constexpr uint32_t GL_RGB10_A2UI = 0x906F;
    static constexpr uint32_t GL_RGB12 = 0x8053;
    static constexpr uint32_t GL_RGB16 = 0x8054;
    static constexpr uint32_t GL_RGB16F = 0x881B;
    static constexpr uint32_t GL_RGB16I = 0x8D89;
    static constexpr uint32_t GL_RGB16UI = 0x8D77;
    static constexpr uint32_t GL_RGB16_SNORM = 0x8F9A;
    static constexpr uint32_t GL_RGB32F = 0x8815;
    static constexpr uint32_t GL_RGB32I = 0x8D83;
    static constexpr uint32_t GL_RGB32UI = 0x8D71;
    static constexpr uint32_t GL_RGB4 = 0x804F;
    static constexpr uint32_t GL_RGB5 = 0x8050;
    static constexpr uint32_t GL_RGB565 = 0x8D62;
    static constexpr uint32_t GL_RGB5_A1 = 0x8057;
    static constexpr uint32_t GL_RGB8 = 0x8051;
    static constexpr uint32_t GL_RGB8I = 0x8D8F;
    static constexpr uint32_t GL_RGB8UI = 0x8D7D;
    static constexpr uint32_t GL_RGB8_SNORM = 0x8F96;
    static constexpr uint32_t GL_RGB9_E5 = 0x8C3D;
    static constexpr uint32_t GL_RGBA = 0x1908;
    static constexpr uint32_t GL_RGBA12 = 0x805A;
    static constexpr uint32_t GL_RGBA16 = 0x805B;
    static constexpr uint32_t GL_RGBA16F = 0x881A;
    static constexpr uint32_t GL_RGBA16I = 0x8D88;
    static constexpr uint32_t GL_RGBA16UI = 0x8D76;
    static constexpr uint32_t GL_RGBA16_SNORM = 0x8F9B;
    static constexpr uint32_t GL_RGBA2 = 0x8055;
    static constexpr uint32_t GL_RGBA32F = 0x8814;
    static constexpr uint32_t GL_RGBA32I = 0x8D82;
    static constexpr uint32_t GL_RGBA32UI = 0x8D70;
    static constexpr uint32_t GL_RGBA4 = 0x8056;
    static constexpr uint32_t GL_RGBA8 = 0x8058;
    static constexpr uint32_t GL_RGBA8I = 0x8D8E;
    static constexpr uint32_t GL_RGBA8UI = 0x8D7C;
    static constexpr uint32_t GL_RGBA8_SNORM = 0x8F97;
    static constexpr uint32_t GL_RGBA_INTEGER = 0x8D99;
    static constexpr uint32_t GL_RGBA_MODE = 0x0C31;
    static constexpr uint32_t GL_RGB_INTEGER = 0x8D98;
    static constexpr uint32_t GL_RGB_SCALE = 0x8573;
    static constexpr uint32_t GL_RG_INTEGER = 0x8228;
    static constexpr uint32_t GL_RIGHT = 0x0407;
    static constexpr uint32_t GL_S = 0x2000;
    static constexpr uint32_t GL_SAMPLER = 0x82E6;
    static constexpr uint32_t GL_SAMPLER_1D = 0x8B5D;
    static constexpr uint32_t GL_SAMPLER_1D_ARRAY = 0x8DC0;
    static constexpr uint32_t GL_SAMPLER_1D_ARRAY_SHADOW = 0x8DC3;
    static constexpr uint32_t GL_SAMPLER_1D_SHADOW = 0x8B61;
    static constexpr uint32_t GL_SAMPLER_2D = 0x8B5E;
    static constexpr uint32_t GL_SAMPLER_2D_ARRAY = 0x8DC1;
    static constexpr uint32_t GL_SAMPLER_2D_ARRAY_SHADOW = 0x8DC4;
    static constexpr uint32_t GL_SAMPLER_2D_MULTISAMPLE = 0x9108;
    static constexpr uint32_t GL_SAMPLER_2D_MULTISAMPLE_ARRAY = 0x910B;
    static constexpr uint32_t GL_SAMPLER_2D_RECT = 0x8B63;
    static constexpr uint32_t GL_SAMPLER_2D_RECT_SHADOW = 0x8B64;
    static constexpr uint32_t GL_SAMPLER_2D_SHADOW = 0x8B62;
    static constexpr uint32_t GL_SAMPLER_3D = 0x8B5F;
    static constexpr uint32_t GL_SAMPLER_BINDING = 0x8919;
    static constexpr uint32_t GL_SAMPLER_BUFFER = 0x8DC2;
    static constexpr uint32_t GL_SAMPLER_CUBE = 0x8B60;
    static constexpr uint32_t GL_SAMPLER_CUBE_MAP_ARRAY = 0x900C;
    static constexpr uint32_t GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW = 0x900D;
    static constexpr uint32_t GL_SAMPLER_CUBE_SHADOW = 0x8DC5;
    static constexpr uint32_t GL_SAMPLES = 0x80A9;
    static constexpr uint32_t GL_SAMPLES_PASSED = 0x8914;
    static constexpr uint32_t GL_SAMPLE_ALPHA_TO_COVERAGE = 0x809E;
    static constexpr uint32_t GL_SAMPLE_ALPHA_TO_ONE = 0x809F;
    static constexpr uint32_t GL_SAMPLE_BUFFERS = 0x80A8;
    static constexpr uint32_t GL_SAMPLE_COVERAGE = 0x80A0;
    static constexpr uint32_t GL_SAMPLE_COVERAGE_INVERT = 0x80AB;
    static constexpr uint32_t GL_SAMPLE_COVERAGE_VALUE = 0x80AA;
    static constexpr uint32_t GL_SAMPLE_MASK = 0x8E51;
    static constexpr uint32_t GL_SAMPLE_MASK_VALUE = 0x8E52;
    static constexpr uint32_t GL_SAMPLE_POSITION = 0x8E50;
    static constexpr uint32_t GL_SAMPLE_SHADING = 0x8C36;
    static constexpr uint32_t GL_SCISSOR_BIT = 0x00080000;
    static constexpr uint32_t GL_SCISSOR_BOX = 0x0C10;
    static constexpr uint32_t GL_SCISSOR_TEST = 0x0C11;
    static constexpr uint32_t GL_SECONDARY_COLOR_ARRAY = 0x845E;
    static constexpr uint32_t GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING = 0x889C;
    static constexpr uint32_t GL_SECONDARY_COLOR_ARRAY_POINTER = 0x845D;
    static constexpr uint32_t GL_SECONDARY_COLOR_ARRAY_SIZE = 0x845A;
    static constexpr uint32_t GL_SECONDARY_COLOR_ARRAY_STRIDE = 0x845C;
    static constexpr uint32_t GL_SECONDARY_COLOR_ARRAY_TYPE = 0x845B;
    static constexpr uint32_t GL_SELECT = 0x1C02;
    static constexpr uint32_t GL_SELECTION_BUFFER_POINTER = 0x0DF3;
    static constexpr uint32_t GL_SELECTION_BUFFER_SIZE = 0x0DF4;
    static constexpr uint32_t GL_SEPARABLE_2D = 0x8012;
    static constexpr uint32_t GL_SEPARATE_ATTRIBS = 0x8C8D;
    static constexpr uint32_t GL_SEPARATE_SPECULAR_COLOR = 0x81FA;
    static constexpr uint32_t GL_SET = 0x150F;
    static constexpr uint32_t GL_SHADER = 0x82E1;
    static constexpr uint32_t GL_SHADER_BINARY_FORMATS = 0x8DF8;
    static constexpr uint32_t GL_SHADER_BINARY_FORMAT_SPIR_V = 0x9551;
    static constexpr uint32_t GL_SHADER_COMPILER = 0x8DFA;
    static constexpr uint32_t GL_SHADER_IMAGE_ACCESS_BARRIER_BIT = 0x00000020;
    static constexpr uint32_t GL_SHADER_IMAGE_ATOMIC = 0x82A6;
    static constexpr uint32_t GL_SHADER_IMAGE_LOAD = 0x82A4;
    static constexpr uint32_t GL_SHADER_IMAGE_STORE = 0x82A5;
    static constexpr uint32_t GL_SHADER_SOURCE_LENGTH = 0x8B88;
    static constexpr uint32_t GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000;
    static constexpr uint32_t GL_SHADER_STORAGE_BLOCK = 0x92E6;
    static constexpr uint32_t GL_SHADER_STORAGE_BUFFER = 0x90D2;
    static constexpr uint32_t GL_SHADER_STORAGE_BUFFER_BINDING = 0x90D3;
    static constexpr uint32_t GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT = 0x90DF;
    static constexpr uint32_t GL_SHADER_STORAGE_BUFFER_SIZE = 0x90D5;
    static constexpr uint32_t GL_SHADER_STORAGE_BUFFER_START = 0x90D4;
    static constexpr uint32_t GL_SHADER_TYPE = 0x8B4F;
    static constexpr uint32_t GL_SHADE_MODEL = 0x0B54;
    static constexpr uint32_t GL_SHADING_LANGUAGE_VERSION = 0x8B8C;
    static constexpr uint32_t GL_SHININESS = 0x1601;
    static constexpr uint32_t GL_SHORT = 0x1402;
    static constexpr uint32_t GL_SIGNALED = 0x9119;
    static constexpr uint32_t GL_SIGNED_NORMALIZED = 0x8F9C;
    static constexpr uint32_t GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST = 0x82AC;
    static constexpr uint32_t GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE = 0x82AE;
    static constexpr uint32_t GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST = 0x82AD;
    static constexpr uint32_t GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE = 0x82AF;
    static constexpr uint32_t GL_SINGLE_COLOR = 0x81F9;
    static constexpr uint32_t GL_SLUMINANCE = 0x8C46;
    static constexpr uint32_t GL_SLUMINANCE8 = 0x8C47;
    static constexpr uint32_t GL_SLUMINANCE8_ALPHA8 = 0x8C45;
    static constexpr uint32_t GL_SLUMINANCE_ALPHA = 0x8C44;
    static constexpr uint32_t GL_SMOOTH = 0x1D01;
    static constexpr uint32_t GL_SMOOTH_LINE_WIDTH_GRANULARITY = 0x0B23;
    static constexpr uint32_t GL_SMOOTH_LINE_WIDTH_RANGE = 0x0B22;
    static constexpr uint32_t GL_SMOOTH_POINT_SIZE_GRANULARITY = 0x0B13;
    static constexpr uint32_t GL_SMOOTH_POINT_SIZE_RANGE = 0x0B12;
    static constexpr uint32_t GL_SOURCE0_ALPHA = 0x8588;
    static constexpr uint32_t GL_SOURCE0_RGB = 0x8580;
    static constexpr uint32_t GL_SOURCE1_ALPHA = 0x8589;
    static constexpr uint32_t GL_SOURCE1_RGB = 0x8581;
    static constexpr uint32_t GL_SOURCE2_ALPHA = 0x858A;
    static constexpr uint32_t GL_SOURCE2_RGB = 0x8582;
    static constexpr uint32_t GL_SPECULAR = 0x1202;
    static constexpr uint32_t GL_SPHERE_MAP = 0x2402;
    static constexpr uint32_t GL_SPIR_V_BINARY = 0x9552;
    static constexpr uint32_t GL_SPIR_V_EXTENSIONS = 0x9553;
    static constexpr uint32_t GL_SPOT_CUTOFF = 0x1206;
    static constexpr uint32_t GL_SPOT_DIRECTION = 0x1204;
    static constexpr uint32_t GL_SPOT_EXPONENT = 0x1205;
    static constexpr uint32_t GL_SRC0_ALPHA = 0x8588;
    static constexpr uint32_t GL_SRC0_RGB = 0x8580;
    static constexpr uint32_t GL_SRC1_ALPHA = 0x8589;
    static constexpr uint32_t GL_SRC1_COLOR = 0x88F9;
    static constexpr uint32_t GL_SRC1_RGB = 0x8581;
    static constexpr uint32_t GL_SRC2_ALPHA = 0x858A;
    static constexpr uint32_t GL_SRC2_RGB = 0x8582;
    static constexpr uint32_t GL_SRC_ALPHA = 0x0302;
    static constexpr uint32_t GL_SRC_ALPHA_SATURATE = 0x0308;
    static constexpr uint32_t GL_SRC_COLOR = 0x0300;
    static constexpr uint32_t GL_SRGB = 0x8C40;
    static constexpr uint32_t GL_SRGB8 = 0x8C41;
    static constexpr uint32_t GL_SRGB8_ALPHA8 = 0x8C43;
    static constexpr uint32_t GL_SRGB_ALPHA = 0x8C42;
    static constexpr uint32_t GL_SRGB_READ = 0x8297;
    static constexpr uint32_t GL_SRGB_WRITE = 0x8298;
    static constexpr uint32_t GL_STACK_OVERFLOW = 0x0503;
    static constexpr uint32_t GL_STACK_UNDERFLOW = 0x0504;
    static constexpr uint32_t GL_STATIC_COPY = 0x88E6;
    static constexpr uint32_t GL_STATIC_DRAW = 0x88E4;
    static constexpr uint32_t GL_STATIC_READ = 0x88E5;
    static constexpr uint32_t GL_STENCIL = 0x1802;
    static constexpr uint32_t GL_STENCIL_ATTACHMENT = 0x8D20;
    static constexpr uint32_t GL_STENCIL_BACK_FAIL = 0x8801;
    static constexpr uint32_t GL_STENCIL_BACK_FUNC = 0x8800;
    static constexpr uint32_t GL_STENCIL_BACK_PASS_DEPTH_FAIL = 0x8802;
    static constexpr uint32_t GL_STENCIL_BACK_PASS_DEPTH_PASS = 0x8803;
    static constexpr uint32_t GL_STENCIL_BACK_REF = 0x8CA3;
    static constexpr uint32_t GL_STENCIL_BACK_VALUE_MASK = 0x8CA4;
    static constexpr uint32_t GL_STENCIL_BACK_WRITEMASK = 0x8CA5;
    static constexpr uint32_t GL_STENCIL_BITS = 0x0D57;
    static constexpr uint32_t GL_STENCIL_BUFFER_BIT = 0x00000400;
    static constexpr uint32_t GL_STENCIL_CLEAR_VALUE = 0x0B91;
    static constexpr uint32_t GL_STENCIL_COMPONENTS = 0x8285;
    static constexpr uint32_t GL_STENCIL_FAIL = 0x0B94;
    static constexpr uint32_t GL_STENCIL_FUNC = 0x0B92;
    static constexpr uint32_t GL_STENCIL_INDEX = 0x1901;
    static constexpr uint32_t GL_STENCIL_INDEX1 = 0x8D46;
    static constexpr uint32_t GL_STENCIL_INDEX16 = 0x8D49;
    static constexpr uint32_t GL_STENCIL_INDEX4 = 0x8D47;
    static constexpr uint32_t GL_STENCIL_INDEX8 = 0x8D48;
    static constexpr uint32_t GL_STENCIL_PASS_DEPTH_FAIL = 0x0B95;
    static constexpr uint32_t GL_STENCIL_PASS_DEPTH_PASS = 0x0B96;
    static constexpr uint32_t GL_STENCIL_REF = 0x0B97;
    static constexpr uint32_t GL_STENCIL_RENDERABLE = 0x8288;
    static constexpr uint32_t GL_STENCIL_TEST = 0x0B90;
    static constexpr uint32_t GL_STENCIL_VALUE_MASK = 0x0B93;
    static constexpr uint32_t GL_STENCIL_WRITEMASK = 0x0B98;
    static constexpr uint32_t GL_STEREO = 0x0C33;
    static constexpr uint32_t GL_STREAM_COPY = 0x88E2;
    static constexpr uint32_t GL_STREAM_DRAW = 0x88E0;
    static constexpr uint32_t GL_STREAM_READ = 0x88E1;
    static constexpr uint32_t GL_SUBPIXEL_BITS = 0x0D50;
    static constexpr uint32_t GL_SUBTRACT = 0x84E7;
    static constexpr uint32_t GL_SYNC_CONDITION = 0x9113;
    static constexpr uint32_t GL_SYNC_FENCE = 0x9116;
    static constexpr uint32_t GL_SYNC_FLAGS = 0x9115;
    static constexpr uint32_t GL_SYNC_FLUSH_COMMANDS_BIT = 0x00000001;
    static constexpr uint32_t GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117;
    static constexpr uint32_t GL_SYNC_STATUS = 0x9114;
    static constexpr uint32_t GL_T = 0x2001;
    static constexpr uint32_t GL_T2F_C3F_V3F = 0x2A2A;
    static constexpr uint32_t GL_T2F_C4F_N3F_V3F = 0x2A2C;
    static constexpr uint32_t GL_T2F_C4UB_V3F = 0x2A29;
    static constexpr uint32_t GL_T2F_N3F_V3F = 0x2A2B;
    static constexpr uint32_t GL_T2F_V3F = 0x2A27;
    static constexpr uint32_t GL_T4F_C4F_N3F_V4F = 0x2A2D;
    static constexpr uint32_t GL_T4F_V4F = 0x2A28;
    static constexpr uint32_t GL_TESS_CONTROL_OUTPUT_VERTICES = 0x8E75;
    static constexpr uint32_t GL_TESS_CONTROL_SHADER = 0x8E88;
    static constexpr uint32_t GL_TESS_CONTROL_SHADER_BIT = 0x00000008;
    static constexpr uint32_t GL_TESS_CONTROL_SHADER_PATCHES = 0x82F1;
    static constexpr uint32_t GL_TESS_CONTROL_SUBROUTINE = 0x92E9;
    static constexpr uint32_t GL_TESS_CONTROL_SUBROUTINE_UNIFORM = 0x92EF;
    static constexpr uint32_t GL_TESS_CONTROL_TEXTURE = 0x829C;
    static constexpr uint32_t GL_TESS_EVALUATION_SHADER = 0x8E87;
    static constexpr uint32_t GL_TESS_EVALUATION_SHADER_BIT = 0x00000010;
    static constexpr uint32_t GL_TESS_EVALUATION_SHADER_INVOCATIONS = 0x82F2;
    static constexpr uint32_t GL_TESS_EVALUATION_SUBROUTINE = 0x92EA;
    static constexpr uint32_t GL_TESS_EVALUATION_SUBROUTINE_UNIFORM = 0x92F0;
    static constexpr uint32_t GL_TESS_EVALUATION_TEXTURE = 0x829D;
    static constexpr uint32_t GL_TESS_GEN_MODE = 0x8E76;
    static constexpr uint32_t GL_TESS_GEN_POINT_MODE = 0x8E79;
    static constexpr uint32_t GL_TESS_GEN_SPACING = 0x8E77;
    static constexpr uint32_t GL_TESS_GEN_VERTEX_ORDER = 0x8E78;
    static constexpr uint32_t GL_TEXTURE = 0x1702;
    static constexpr uint32_t GL_TEXTURE0 = 0x84C0;
    static constexpr uint32_t GL_TEXTURE1 = 0x84C1;
    static constexpr uint32_t GL_TEXTURE10 = 0x84CA;
    static constexpr uint32_t GL_TEXTURE11 = 0x84CB;
    static constexpr uint32_t GL_TEXTURE12 = 0x84CC;
    static constexpr uint32_t GL_TEXTURE13 = 0x84CD;
    static constexpr uint32_t GL_TEXTURE14 = 0x84CE;
    static constexpr uint32_t GL_TEXTURE15 = 0x84CF;
    static constexpr uint32_t GL_TEXTURE16 = 0x84D0;
    static constexpr uint32_t GL_TEXTURE17 = 0x84D1;
    static constexpr uint32_t GL_TEXTURE18 = 0x84D2;
    static constexpr uint32_t GL_TEXTURE19 = 0x84D3;
    static constexpr uint32_t GL_TEXTURE2 = 0x84C2;
    static constexpr uint32_t GL_TEXTURE20 = 0x84D4;
    static constexpr uint32_t GL_TEXTURE21 = 0x84D5;
    static constexpr uint32_t GL_TEXTURE22 = 0x84D6;
    static constexpr uint32_t GL_TEXTURE23 = 0x84D7;
    static constexpr uint32_t GL_TEXTURE24 = 0x84D8;
    static constexpr uint32_t GL_TEXTURE25 = 0x84D9;
    static constexpr uint32_t GL_TEXTURE26 = 0x84DA;
    static constexpr uint32_t GL_TEXTURE27 = 0x84DB;
    static constexpr uint32_t GL_TEXTURE28 = 0x84DC;
    static constexpr uint32_t GL_TEXTURE29 = 0x84DD;
    static constexpr uint32_t GL_TEXTURE3 = 0x84C3;
    static constexpr uint32_t GL_TEXTURE30 = 0x84DE;
    static constexpr uint32_t GL_TEXTURE31 = 0x84DF;
    static constexpr uint32_t GL_TEXTURE4 = 0x84C4;
    static constexpr uint32_t GL_TEXTURE5 = 0x84C5;
    static constexpr uint32_t GL_TEXTURE6 = 0x84C6;
    static constexpr uint32_t GL_TEXTURE7 = 0x84C7;
    static constexpr uint32_t GL_TEXTURE8 = 0x84C8;
    static constexpr uint32_t GL_TEXTURE9 = 0x84C9;
    static constexpr uint32_t GL_TEXTURE_1D = 0x0DE0;
    static constexpr uint32_t GL_TEXTURE_1D_ARRAY = 0x8C18;
    static constexpr uint32_t GL_TEXTURE_2D = 0x0DE1;
    static constexpr uint32_t GL_TEXTURE_2D_ARRAY = 0x8C1A;
    static constexpr uint32_t GL_TEXTURE_2D_MULTISAMPLE = 0x9100;
    static constexpr uint32_t GL_TEXTURE_2D_MULTISAMPLE_ARRAY = 0x9102;
    static constexpr uint32_t GL_TEXTURE_3D = 0x806F;
    static constexpr uint32_t GL_TEXTURE_ALPHA_SIZE = 0x805F;
    static constexpr uint32_t GL_TEXTURE_ALPHA_TYPE = 0x8C13;
    static constexpr uint32_t GL_TEXTURE_BASE_LEVEL = 0x813C;
    static constexpr uint32_t GL_TEXTURE_BINDING_1D = 0x8068;
    static constexpr uint32_t GL_TEXTURE_BINDING_1D_ARRAY = 0x8C1C;
    static constexpr uint32_t GL_TEXTURE_BINDING_2D = 0x8069;
    static constexpr uint32_t GL_TEXTURE_BINDING_2D_ARRAY = 0x8C1D;
    static constexpr uint32_t GL_TEXTURE_BINDING_2D_MULTISAMPLE = 0x9104;
    static constexpr uint32_t GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY = 0x9105;
    static constexpr uint32_t GL_TEXTURE_BINDING_3D = 0x806A;
    static constexpr uint32_t GL_TEXTURE_BINDING_BUFFER = 0x8C2C;
    static constexpr uint32_t GL_TEXTURE_BINDING_CUBE_MAP = 0x8514;
    static constexpr uint32_t GL_TEXTURE_BINDING_CUBE_MAP_ARRAY = 0x900A;
    static constexpr uint32_t GL_TEXTURE_BINDING_RECTANGLE = 0x84F6;
    static constexpr uint32_t GL_TEXTURE_BIT = 0x00040000;
    static constexpr uint32_t GL_TEXTURE_BLUE_SIZE = 0x805E;
    static constexpr uint32_t GL_TEXTURE_BLUE_TYPE = 0x8C12;
    static constexpr uint32_t GL_TEXTURE_BORDER = 0x1005;
    static constexpr uint32_t GL_TEXTURE_BORDER_COLOR = 0x1004;
    static constexpr uint32_t GL_TEXTURE_BUFFER = 0x8C2A;
    static constexpr uint32_t GL_TEXTURE_BUFFER_BINDING = 0x8C2A;
    static constexpr uint32_t GL_TEXTURE_BUFFER_DATA_STORE_BINDING = 0x8C2D;
    static constexpr uint32_t GL_TEXTURE_BUFFER_OFFSET = 0x919D;
    static constexpr uint32_t GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT = 0x919F;
    static constexpr uint32_t GL_TEXTURE_BUFFER_SIZE = 0x919E;
    static constexpr uint32_t GL_TEXTURE_COMPARE_FUNC = 0x884D;
    static constexpr uint32_t GL_TEXTURE_COMPARE_MODE = 0x884C;
    static constexpr uint32_t GL_TEXTURE_COMPONENTS = 0x1003;
    static constexpr uint32_t GL_TEXTURE_COMPRESSED = 0x86A1;
    static constexpr uint32_t GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT = 0x82B2;
    static constexpr uint32_t GL_TEXTURE_COMPRESSED_BLOCK_SIZE = 0x82B3;
    static constexpr uint32_t GL_TEXTURE_COMPRESSED_BLOCK_WIDTH = 0x82B1;
    static constexpr uint32_t GL_TEXTURE_COMPRESSED_IMAGE_SIZE = 0x86A0;
    static constexpr uint32_t GL_TEXTURE_COMPRESSION_HINT = 0x84EF;
    static constexpr uint32_t GL_TEXTURE_COORD_ARRAY = 0x8078;
    static constexpr uint32_t GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING = 0x889A;
    static constexpr uint32_t GL_TEXTURE_COORD_ARRAY_POINTER = 0x8092;
    static constexpr uint32_t GL_TEXTURE_COORD_ARRAY_SIZE = 0x8088;
    static constexpr uint32_t GL_TEXTURE_COORD_ARRAY_STRIDE = 0x808A;
    static constexpr uint32_t GL_TEXTURE_COORD_ARRAY_TYPE = 0x8089;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP = 0x8513;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_ARRAY = 0x9009;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_NEGATIVE_X = 0x8516;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_NEGATIVE_Y = 0x8518;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_NEGATIVE_Z = 0x851A;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_POSITIVE_Y = 0x8517;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_POSITIVE_Z = 0x8519;
    static constexpr uint32_t GL_TEXTURE_CUBE_MAP_SEAMLESS = 0x884F;
    static constexpr uint32_t GL_TEXTURE_DEPTH = 0x8071;
    static constexpr uint32_t GL_TEXTURE_DEPTH_SIZE = 0x884A;
    static constexpr uint32_t GL_TEXTURE_DEPTH_TYPE = 0x8C16;
    static constexpr uint32_t GL_TEXTURE_ENV = 0x2300;
    static constexpr uint32_t GL_TEXTURE_ENV_COLOR = 0x2201;
    static constexpr uint32_t GL_TEXTURE_ENV_MODE = 0x2200;
    static constexpr uint32_t GL_TEXTURE_FETCH_BARRIER_BIT = 0x00000008;
    static constexpr uint32_t GL_TEXTURE_FILTER_CONTROL = 0x8500;
    static constexpr uint32_t GL_TEXTURE_FIXED_SAMPLE_LOCATIONS = 0x9107;
    static constexpr uint32_t GL_TEXTURE_GATHER = 0x82A2;
    static constexpr uint32_t GL_TEXTURE_GATHER_SHADOW = 0x82A3;
    static constexpr uint32_t GL_TEXTURE_GEN_MODE = 0x2500;
    static constexpr uint32_t GL_TEXTURE_GEN_Q = 0x0C63;
    static constexpr uint32_t GL_TEXTURE_GEN_R = 0x0C62;
    static constexpr uint32_t GL_TEXTURE_GEN_S = 0x0C60;
    static constexpr uint32_t GL_TEXTURE_GEN_T = 0x0C61;
    static constexpr uint32_t GL_TEXTURE_GREEN_SIZE = 0x805D;
    static constexpr uint32_t GL_TEXTURE_GREEN_TYPE = 0x8C11;
    static constexpr uint32_t GL_TEXTURE_HEIGHT = 0x1001;
    static constexpr uint32_t GL_TEXTURE_IMAGE_FORMAT = 0x828F;
    static constexpr uint32_t GL_TEXTURE_IMAGE_TYPE = 0x8290;
    static constexpr uint32_t GL_TEXTURE_IMMUTABLE_FORMAT = 0x912F;
    static constexpr uint32_t GL_TEXTURE_IMMUTABLE_LEVELS = 0x82DF;
    static constexpr uint32_t GL_TEXTURE_INTENSITY_SIZE = 0x8061;
    static constexpr uint32_t GL_TEXTURE_INTENSITY_TYPE = 0x8C15;
    static constexpr uint32_t GL_TEXTURE_INTERNAL_FORMAT = 0x1003;
    static constexpr uint32_t GL_TEXTURE_LOD_BIAS = 0x8501;
    static constexpr uint32_t GL_TEXTURE_LUMINANCE_SIZE = 0x8060;
    static constexpr uint32_t GL_TEXTURE_LUMINANCE_TYPE = 0x8C14;
    static constexpr uint32_t GL_TEXTURE_MAG_FILTER = 0x2800;
    static constexpr uint32_t GL_TEXTURE_MATRIX = 0x0BA8;
    static constexpr uint32_t GL_TEXTURE_MAX_ANISOTROPY = 0x84FE;
    static constexpr uint32_t GL_TEXTURE_MAX_LEVEL = 0x813D;
    static constexpr uint32_t GL_TEXTURE_MAX_LOD = 0x813B;
    static constexpr uint32_t GL_TEXTURE_MIN_FILTER = 0x2801;
    static constexpr uint32_t GL_TEXTURE_MIN_LOD = 0x813A;
    static constexpr uint32_t GL_TEXTURE_PRIORITY = 0x8066;
    static constexpr uint32_t GL_TEXTURE_RECTANGLE = 0x84F5;
    static constexpr uint32_t GL_TEXTURE_RED_SIZE = 0x805C;
    static constexpr uint32_t GL_TEXTURE_RED_TYPE = 0x8C10;
    static constexpr uint32_t GL_TEXTURE_RESIDENT = 0x8067;
    static constexpr uint32_t GL_TEXTURE_SAMPLES = 0x9106;
    static constexpr uint32_t GL_TEXTURE_SHADOW = 0x82A1;
    static constexpr uint32_t GL_TEXTURE_SHARED_SIZE = 0x8C3F;
    static constexpr uint32_t GL_TEXTURE_STACK_DEPTH = 0x0BA5;
    static constexpr uint32_t GL_TEXTURE_STENCIL_SIZE = 0x88F1;
    static constexpr uint32_t GL_TEXTURE_SWIZZLE_A = 0x8E45;
    static constexpr uint32_t GL_TEXTURE_SWIZZLE_B = 0x8E44;
    static constexpr uint32_t GL_TEXTURE_SWIZZLE_G = 0x8E43;
    static constexpr uint32_t GL_TEXTURE_SWIZZLE_R = 0x8E42;
    static constexpr uint32_t GL_TEXTURE_SWIZZLE_RGBA = 0x8E46;
    static constexpr uint32_t GL_TEXTURE_TARGET = 0x1006;
    static constexpr uint32_t GL_TEXTURE_UPDATE_BARRIER_BIT = 0x00000100;
    static constexpr uint32_t GL_TEXTURE_VIEW = 0x82B5;
    static constexpr uint32_t GL_TEXTURE_VIEW_MIN_LAYER = 0x82DD;
    static constexpr uint32_t GL_TEXTURE_VIEW_MIN_LEVEL = 0x82DB;
    static constexpr uint32_t GL_TEXTURE_VIEW_NUM_LAYERS = 0x82DE;
    static constexpr uint32_t GL_TEXTURE_VIEW_NUM_LEVELS = 0x82DC;
    static constexpr uint32_t GL_TEXTURE_WIDTH = 0x1000;
    static constexpr uint32_t GL_TEXTURE_WRAP_R = 0x8072;
    static constexpr uint32_t GL_TEXTURE_WRAP_S = 0x2802;
    static constexpr uint32_t GL_TEXTURE_WRAP_T = 0x2803;
    static constexpr uint32_t GL_TIMEOUT_EXPIRED = 0x911B;
    static constexpr uint64_t GL_TIMEOUT_IGNORED = 0xFFFFFFFFFFFFFFFF;
    static constexpr uint32_t GL_TIMESTAMP = 0x8E28;
    static constexpr uint32_t GL_TIME_ELAPSED = 0x88BF;
    static constexpr uint32_t GL_TOP_LEVEL_ARRAY_SIZE = 0x930C;
    static constexpr uint32_t GL_TOP_LEVEL_ARRAY_STRIDE = 0x930D;
    static constexpr uint32_t GL_TRANSFORM_BIT = 0x00001000;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK = 0x8E22;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_ACTIVE = 0x8E24;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BARRIER_BIT = 0x00000800;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BINDING = 0x8E25;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER = 0x8C8E;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_ACTIVE = 0x8E24;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_BINDING = 0x8C8F;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_INDEX = 0x934B;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_MODE = 0x8C7F;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_PAUSED = 0x8E23;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_SIZE = 0x8C85;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_START = 0x8C84;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE = 0x934C;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_OVERFLOW = 0x82EC;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_PAUSED = 0x8E23;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN = 0x8C88;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW = 0x82ED;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_VARYING = 0x92F4;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_VARYINGS = 0x8C83;
    static constexpr uint32_t GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH = 0x8C76;
    static constexpr uint32_t GL_TRANSPOSE_COLOR_MATRIX = 0x84E6;
    static constexpr uint32_t GL_TRANSPOSE_MODELVIEW_MATRIX = 0x84E3;
    static constexpr uint32_t GL_TRANSPOSE_PROJECTION_MATRIX = 0x84E4;
    static constexpr uint32_t GL_TRANSPOSE_TEXTURE_MATRIX = 0x84E5;
    static constexpr uint32_t GL_TRIANGLES = 0x0004;
    static constexpr uint32_t GL_TRIANGLES_ADJACENCY = 0x000C;
    static constexpr uint32_t GL_TRIANGLE_FAN = 0x0006;
    static constexpr uint32_t GL_TRIANGLE_STRIP = 0x0005;
    static constexpr uint32_t GL_TRIANGLE_STRIP_ADJACENCY = 0x000D;
    static constexpr uint32_t GL_TRUE = 1;
    static constexpr uint32_t GL_TYPE = 0x92FA;
    static constexpr uint32_t GL_UNDEFINED_VERTEX = 0x8260;
    static constexpr uint32_t GL_UNIFORM = 0x92E1;
    static constexpr uint32_t GL_UNIFORM_ARRAY_STRIDE = 0x8A3C;
    static constexpr uint32_t GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX = 0x92DA;
    static constexpr uint32_t GL_UNIFORM_BARRIER_BIT = 0x00000004;
    static constexpr uint32_t GL_UNIFORM_BLOCK = 0x92E2;
    static constexpr uint32_t GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS = 0x8A42;
    static constexpr uint32_t GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES = 0x8A43;
    static constexpr uint32_t GL_UNIFORM_BLOCK_BINDING = 0x8A3F;
    static constexpr uint32_t GL_UNIFORM_BLOCK_DATA_SIZE = 0x8A40;
    static constexpr uint32_t GL_UNIFORM_BLOCK_INDEX = 0x8A3A;
    static constexpr uint32_t GL_UNIFORM_BLOCK_NAME_LENGTH = 0x8A41;
    static constexpr uint32_t GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER = 0x90EC;
    static constexpr uint32_t GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER = 0x8A46;
    static constexpr uint32_t GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER = 0x8A45;
    static constexpr uint32_t GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER = 0x84F0;
    static constexpr uint32_t GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER = 0x84F1;
    static constexpr uint32_t GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER = 0x8A44;
    static constexpr uint32_t GL_UNIFORM_BUFFER = 0x8A11;
    static constexpr uint32_t GL_UNIFORM_BUFFER_BINDING = 0x8A28;
    static constexpr uint32_t GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT = 0x8A34;
    static constexpr uint32_t GL_UNIFORM_BUFFER_SIZE = 0x8A2A;
    static constexpr uint32_t GL_UNIFORM_BUFFER_START = 0x8A29;
    static constexpr uint32_t GL_UNIFORM_IS_ROW_MAJOR = 0x8A3E;
    static constexpr uint32_t GL_UNIFORM_MATRIX_STRIDE = 0x8A3D;
    static constexpr uint32_t GL_UNIFORM_NAME_LENGTH = 0x8A39;
    static constexpr uint32_t GL_UNIFORM_OFFSET = 0x8A3B;
    static constexpr uint32_t GL_UNIFORM_SIZE = 0x8A38;
    static constexpr uint32_t GL_UNIFORM_TYPE = 0x8A37;
    static constexpr uint32_t GL_UNKNOWN_CONTEXT_RESET = 0x8255;
    static constexpr uint32_t GL_UNPACK_ALIGNMENT = 0x0CF5;
    static constexpr uint32_t GL_UNPACK_COMPRESSED_BLOCK_DEPTH = 0x9129;
    static constexpr uint32_t GL_UNPACK_COMPRESSED_BLOCK_HEIGHT = 0x9128;
    static constexpr uint32_t GL_UNPACK_COMPRESSED_BLOCK_SIZE = 0x912A;
    static constexpr uint32_t GL_UNPACK_COMPRESSED_BLOCK_WIDTH = 0x9127;
    static constexpr uint32_t GL_UNPACK_IMAGE_HEIGHT = 0x806E;
    static constexpr uint32_t GL_UNPACK_LSB_FIRST = 0x0CF1;
    static constexpr uint32_t GL_UNPACK_ROW_LENGTH = 0x0CF2;
    static constexpr uint32_t GL_UNPACK_SKIP_IMAGES = 0x806D;
    static constexpr uint32_t GL_UNPACK_SKIP_PIXELS = 0x0CF4;
    static constexpr uint32_t GL_UNPACK_SKIP_ROWS = 0x0CF3;
    static constexpr uint32_t GL_UNPACK_SWAP_BYTES = 0x0CF0;
    static constexpr uint32_t GL_UNSIGNALED = 0x9118;
    static constexpr uint32_t GL_UNSIGNED_BYTE = 0x1401;
    static constexpr uint32_t GL_UNSIGNED_BYTE_2_3_3_REV = 0x8362;
    static constexpr uint32_t GL_UNSIGNED_BYTE_3_3_2 = 0x8032;
    static constexpr uint32_t GL_UNSIGNED_INT = 0x1405;
    static constexpr uint32_t GL_UNSIGNED_INT_10F_11F_11F_REV = 0x8C3B;
    static constexpr uint32_t GL_UNSIGNED_INT_10_10_10_2 = 0x8036;
    static constexpr uint32_t GL_UNSIGNED_INT_24_8 = 0x84FA;
    static constexpr uint32_t GL_UNSIGNED_INT_2_10_10_10_REV = 0x8368;
    static constexpr uint32_t GL_UNSIGNED_INT_5_9_9_9_REV = 0x8C3E;
    static constexpr uint32_t GL_UNSIGNED_INT_8_8_8_8 = 0x8035;
    static constexpr uint32_t GL_UNSIGNED_INT_8_8_8_8_REV = 0x8367;
    static constexpr uint32_t GL_UNSIGNED_INT_ATOMIC_COUNTER = 0x92DB;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_1D = 0x9062;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_1D_ARRAY = 0x9068;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_2D = 0x9063;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_2D_ARRAY = 0x9069;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE = 0x906B;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY = 0x906C;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_2D_RECT = 0x9065;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_3D = 0x9064;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_BUFFER = 0x9067;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_CUBE = 0x9066;
    static constexpr uint32_t GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY = 0x906A;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_1D = 0x8DD1;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_1D_ARRAY = 0x8DD6;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_2D = 0x8DD2;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_2D_ARRAY = 0x8DD7;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE = 0x910A;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY = 0x910D;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_2D_RECT = 0x8DD5;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_3D = 0x8DD3;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_BUFFER = 0x8DD8;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_CUBE = 0x8DD4;
    static constexpr uint32_t GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY = 0x900F;
    static constexpr uint32_t GL_UNSIGNED_INT_VEC2 = 0x8DC6;
    static constexpr uint32_t GL_UNSIGNED_INT_VEC3 = 0x8DC7;
    static constexpr uint32_t GL_UNSIGNED_INT_VEC4 = 0x8DC8;
    static constexpr uint32_t GL_UNSIGNED_NORMALIZED = 0x8C17;
    static constexpr uint32_t GL_UNSIGNED_SHORT = 0x1403;
    static constexpr uint32_t GL_UNSIGNED_SHORT_1_5_5_5_REV = 0x8366;
    static constexpr uint32_t GL_UNSIGNED_SHORT_4_4_4_4 = 0x8033;
    static constexpr uint32_t GL_UNSIGNED_SHORT_4_4_4_4_REV = 0x8365;
    static constexpr uint32_t GL_UNSIGNED_SHORT_5_5_5_1 = 0x8034;
    static constexpr uint32_t GL_UNSIGNED_SHORT_5_6_5 = 0x8363;
    static constexpr uint32_t GL_UNSIGNED_SHORT_5_6_5_REV = 0x8364;
    static constexpr uint32_t GL_UPPER_LEFT = 0x8CA2;
    static constexpr uint32_t GL_V2F = 0x2A20;
    static constexpr uint32_t GL_V3F = 0x2A21;
    static constexpr uint32_t GL_VALIDATE_STATUS = 0x8B83;
    static constexpr uint32_t GL_VENDOR = 0x1F00;
    static constexpr uint32_t GL_VERSION = 0x1F02;
    static constexpr uint32_t GL_VERTEX_ARRAY = 0x8074;
    static constexpr uint32_t GL_VERTEX_ARRAY_BINDING = 0x85B5;
    static constexpr uint32_t GL_VERTEX_ARRAY_BUFFER_BINDING = 0x8896;
    static constexpr uint32_t GL_VERTEX_ARRAY_POINTER = 0x808E;
    static constexpr uint32_t GL_VERTEX_ARRAY_SIZE = 0x807A;
    static constexpr uint32_t GL_VERTEX_ARRAY_STRIDE = 0x807C;
    static constexpr uint32_t GL_VERTEX_ARRAY_TYPE = 0x807B;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT = 0x00000001;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING = 0x889F;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_DIVISOR = 0x88FE;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_ENABLED = 0x8622;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_INTEGER = 0x88FD;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_LONG = 0x874E;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_NORMALIZED = 0x886A;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_POINTER = 0x8645;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_SIZE = 0x8623;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_STRIDE = 0x8624;
    static constexpr uint32_t GL_VERTEX_ATTRIB_ARRAY_TYPE = 0x8625;
    static constexpr uint32_t GL_VERTEX_ATTRIB_BINDING = 0x82D4;
    static constexpr uint32_t GL_VERTEX_ATTRIB_RELATIVE_OFFSET = 0x82D5;
    static constexpr uint32_t GL_VERTEX_BINDING_BUFFER = 0x8F4F;
    static constexpr uint32_t GL_VERTEX_BINDING_DIVISOR = 0x82D6;
    static constexpr uint32_t GL_VERTEX_BINDING_OFFSET = 0x82D7;
    static constexpr uint32_t GL_VERTEX_BINDING_STRIDE = 0x82D8;
    static constexpr uint32_t GL_VERTEX_PROGRAM_POINT_SIZE = 0x8642;
    static constexpr uint32_t GL_VERTEX_PROGRAM_TWO_SIDE = 0x8643;
    static constexpr uint32_t GL_VERTEX_SHADER = 0x8B31;
    static constexpr uint32_t GL_VERTEX_SHADER_BIT = 0x00000001;
    static constexpr uint32_t GL_VERTEX_SHADER_INVOCATIONS = 0x82F0;
    static constexpr uint32_t GL_VERTEX_SUBROUTINE = 0x92E8;
    static constexpr uint32_t GL_VERTEX_SUBROUTINE_UNIFORM = 0x92EE;
    static constexpr uint32_t GL_VERTEX_TEXTURE = 0x829B;
    static constexpr uint32_t GL_VERTICES_SUBMITTED = 0x82EE;
    static constexpr uint32_t GL_VIEWPORT = 0x0BA2;
    static constexpr uint32_t GL_VIEWPORT_BIT = 0x00000800;
    static constexpr uint32_t GL_VIEWPORT_BOUNDS_RANGE = 0x825D;
    static constexpr uint32_t GL_VIEWPORT_INDEX_PROVOKING_VERTEX = 0x825F;
    static constexpr uint32_t GL_VIEWPORT_SUBPIXEL_BITS = 0x825C;
    static constexpr uint32_t GL_VIEW_CLASS_128_BITS = 0x82C4;
    static constexpr uint32_t GL_VIEW_CLASS_16_BITS = 0x82CA;
    static constexpr uint32_t GL_VIEW_CLASS_24_BITS = 0x82C9;
    static constexpr uint32_t GL_VIEW_CLASS_32_BITS = 0x82C8;
    static constexpr uint32_t GL_VIEW_CLASS_48_BITS = 0x82C7;
    static constexpr uint32_t GL_VIEW_CLASS_64_BITS = 0x82C6;
    static constexpr uint32_t GL_VIEW_CLASS_8_BITS = 0x82CB;
    static constexpr uint32_t GL_VIEW_CLASS_96_BITS = 0x82C5;
    static constexpr uint32_t GL_VIEW_CLASS_BPTC_FLOAT = 0x82D3;
    static constexpr uint32_t GL_VIEW_CLASS_BPTC_UNORM = 0x82D2;
    static constexpr uint32_t GL_VIEW_CLASS_RGTC1_RED = 0x82D0;
    static constexpr uint32_t GL_VIEW_CLASS_RGTC2_RG = 0x82D1;
    static constexpr uint32_t GL_VIEW_CLASS_S3TC_DXT1_RGB = 0x82CC;
    static constexpr uint32_t GL_VIEW_CLASS_S3TC_DXT1_RGBA = 0x82CD;
    static constexpr uint32_t GL_VIEW_CLASS_S3TC_DXT3_RGBA = 0x82CE;
    static constexpr uint32_t GL_VIEW_CLASS_S3TC_DXT5_RGBA = 0x82CF;
    static constexpr uint32_t GL_VIEW_COMPATIBILITY_CLASS = 0x82B6;
    static constexpr uint32_t GL_WAIT_FAILED = 0x911D;
    static constexpr uint32_t GL_WEIGHT_ARRAY_BUFFER_BINDING = 0x889E;
    static constexpr uint32_t GL_WRITE_ONLY = 0x88B9;
    static constexpr uint32_t GL_XOR = 0x1506;
    static constexpr uint32_t GL_ZERO = 0;
    static constexpr uint32_t GL_ZERO_TO_ONE = 0x935F;
    static constexpr uint32_t GL_ZOOM_X = 0x0D16;
    static constexpr uint32_t GL_ZOOM_Y = 0x0D17;

    // khronos types +
    #ifndef __khrplatform_h_
    #define __khrplatform_h_

    /*
    ** Copyright (c) 2008-2018 The Khronos Group Inc.
    **
    ** Permission is hereby granted, free of charge, to any person obtaining a
    ** copy of this software and/or associated documentation files (the
    ** "Materials"), to deal in the Materials without restriction, including
    ** without limitation the rights to use, copy, modify, merge, publish,
    ** distribute, sublicense, and/or sell copies of the Materials, and to
    ** permit persons to whom the Materials are furnished to do so, subject to
    ** the following conditions:
    **
    ** The above copyright notice and this permission notice shall be included
    ** in all copies or substantial portions of the Materials.
    **
    ** THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    ** EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    ** MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    ** IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    ** CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    ** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    ** MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
    */

    /* Khronos platform-specific types and definitions.
     *
     * The master copy of khrplatform.h is maintained in the Khronos EGL
     * Registry repository at https://github.com/KhronosGroup/EGL-Registry
     * The last semantic modification to khrplatform.h was at commit ID:
     *      67a3e0864c2d75ea5287b9f3d2eb74a745936692
     *
     * Adopters may modify this file to suit their platform. Adopters are
     * encouraged to submit platform specific modifications to the Khronos
     * group so that they can be included in future versions of this file.
     * Please submit changes by filing pull requests or issues on
     * the EGL Registry repository linked above.
     *
     *
     * See the Implementer's Guidelines for information about where this file
     * should be located on your system and for more details of its use:
     *    http://www.khronos.org/registry/implementers_guide.pdf
     *
     * This file should be included as
     *        #include <KHR/khrplatform.h>
     * by Khronos client API header files that use its types and defines.
     *
     * The types in khrplatform.h should only be used to define API-specific types.
     *
     * Types defined in khrplatform.h:
     *    khronos_int8_t              signed   8  bit
     *    khronos_uint8_t             unsigned 8  bit
     *    khronos_int16_t             signed   16 bit
     *    khronos_uint16_t            unsigned 16 bit
     *    khronos_int32_t             signed   32 bit
     *    khronos_uint32_t            unsigned 32 bit
     *    khronos_int64_t             signed   64 bit
     *    khronos_uint64_t            unsigned 64 bit
     *    khronos_intptr_t            signed   same number of bits as a pointer
     *    khronos_uintptr_t           unsigned same number of bits as a pointer
     *    khronos_ssize_t             signed   size
     *    khronos_usize_t             unsigned size
     *    khronos_float_t             signed   32 bit floating point
     *    khronos_time_ns_t           unsigned 64 bit time in nanoseconds
     *    khronos_utime_nanoseconds_t unsigned time interval or absolute time in
     *                                         nanoseconds
     *    khronos_stime_nanoseconds_t signed time interval in nanoseconds
     *    khronos_boolean_enum_t      enumerated boolean type. This should
     *      only be used as a base type when a client API's boolean type is
     *      an enum. Client APIs which use an integer or other type for
     *      booleans cannot use this as the base type for their boolean.
     *
     * Tokens defined in khrplatform.h:
     *
     *    KHRONOS_FALSE, KHRONOS_TRUE Enumerated boolean false/true values.
     *
     *    KHRONOS_SUPPORT_INT64 is 1 if 64 bit integers are supported; otherwise 0.
     *    KHRONOS_SUPPORT_FLOAT is 1 if floats are supported; otherwise 0.
     *
     * Calling convention macros defined in this file:
     *    KHRONOS_APICALL
     *    KHRONOS_APIENTRY
     *    KHRONOS_APIATTRIBUTES
     *
     * These may be used in function prototypes as:
     *
     *      KHRONOS_APICALL void KHRONOS_APIENTRY funcname(
     *                                  int arg1,
     *                                  int arg2) KHRONOS_APIATTRIBUTES;
     */

    #if defined(__SCITECH_SNAP__) && !defined(KHRONOS_STATIC)
    #   define KHRONOS_STATIC 1
    #endif

     /*-------------------------------------------------------------------------
      * Definition of KHRONOS_APICALL
      *-------------------------------------------------------------------------
      * This precedes the return type of the function in the function prototype.
      */
    #if defined(KHRONOS_STATIC)
      /* If the preprocessor constant KHRONOS_STATIC is defined, make the
       * header compatible with static linking. */
    #   define KHRONOS_APICALL
    #elif defined(_WIN32)
    #   define KHRONOS_APICALL __declspec(dllimport)
    #elif defined (__SYMBIAN32__)
    #   define KHRONOS_APICALL IMPORT_C
    #elif defined(__ANDROID__)
    #   define KHRONOS_APICALL __attribute__((visibility("default")))
    #else
    #   define KHRONOS_APICALL
    #endif

      /*-------------------------------------------------------------------------
       * Definition of KHRONOS_APIENTRY
       *-------------------------------------------------------------------------
       * This follows the return type of the function  and precedes the function
       * name in the function prototype.
       */
    #if defined(_WIN32) && !defined(_WIN32_WCE) && !defined(__SCITECH_SNAP__)
       /* Win32 but not WinCE */
    #   define KHRONOS_APIENTRY __stdcall
    #else
    #   define KHRONOS_APIENTRY
    #endif

       /*-------------------------------------------------------------------------
        * Definition of KHRONOS_APIATTRIBUTES
        *-------------------------------------------------------------------------
        * This follows the closing parenthesis of the function prototype arguments.
        */
    #if defined (__ARMCC_2__)
    #define KHRONOS_APIATTRIBUTES __softfp
    #else
    #define KHRONOS_APIATTRIBUTES
    #endif

        /*-------------------------------------------------------------------------
         * basic type definitions
         *-----------------------------------------------------------------------*/
    #if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) || defined(__GNUC__) || defined(__SCO__) || defined(__USLC__)


         /*
          * Using <stdint.h>
          */
    #include <stdint.h>
    typedef int32_t                 khronos_int32_t;
    typedef uint32_t                khronos_uint32_t;
    typedef int64_t                 khronos_int64_t;
    typedef uint64_t                khronos_uint64_t;
    #define KHRONOS_SUPPORT_INT64   1
    #define KHRONOS_SUPPORT_FLOAT   1
    /*
     * To support platform where unsigned long cannot be used interchangeably with
     * inptr_t (e.g. CHERI-extended ISAs), we can use the stdint.h intptr_t.
     * Ideally, we could just use (u)intptr_t everywhere, but this could result in
     * ABI breakage if khronos_uintptr_t is changed from unsigned long to
     * unsigned long long or similar (this results in different C++ name mangling).
     * To avoid changes for existing platforms, we restrict usage of intptr_t to
     * platforms where the size of a pointer is larger than the size of long.
     */
    #if defined(__SIZEOF_LONG__) && defined(__SIZEOF_POINTER__)
    #if __SIZEOF_POINTER__ > __SIZEOF_LONG__
    #define KHRONOS_USE_INTPTR_T
    #endif
    #endif

    #elif defined(__VMS ) || defined(__sgi)

         /*
          * Using <inttypes.h>
          */
    #include <inttypes.h>
    typedef int32_t                 khronos_int32_t;
    typedef uint32_t                khronos_uint32_t;
    typedef int64_t                 khronos_int64_t;
    typedef uint64_t                khronos_uint64_t;
    #define KHRONOS_SUPPORT_INT64   1
    #define KHRONOS_SUPPORT_FLOAT   1

    #elif defined(_WIN32) && !defined(__SCITECH_SNAP__)

         /*
          * Win32
          */
    typedef __int32                 khronos_int32_t;
    typedef unsigned __int32        khronos_uint32_t;
    typedef __int64                 khronos_int64_t;
    typedef unsigned __int64        khronos_uint64_t;
    #define KHRONOS_SUPPORT_INT64   1
    #define KHRONOS_SUPPORT_FLOAT   1

    #elif defined(__sun__) || defined(__digital__)

         /*
          * Sun or Digital
          */
    typedef int                     khronos_int32_t;
    typedef unsigned int            khronos_uint32_t;
    #if defined(__arch64__) || defined(_LP64)
    typedef long int                khronos_int64_t;
    typedef unsigned long int       khronos_uint64_t;
    #else
    typedef long long int           khronos_int64_t;
    typedef unsigned long long int  khronos_uint64_t;
    #endif /* __arch64__ */
    #define KHRONOS_SUPPORT_INT64   1
    #define KHRONOS_SUPPORT_FLOAT   1

    #elif 0

         /*
          * Hypothetical platform with no float or int64 support
          */
    typedef int                     khronos_int32_t;
    typedef unsigned int            khronos_uint32_t;
    #define KHRONOS_SUPPORT_INT64   0
    #define KHRONOS_SUPPORT_FLOAT   0

    #else

         /*
          * Generic fallback
          */
    #include <stdint.h>
    typedef int32_t                 khronos_int32_t;
    typedef uint32_t                khronos_uint32_t;
    typedef int64_t                 khronos_int64_t;
    typedef uint64_t                khronos_uint64_t;
    #define KHRONOS_SUPPORT_INT64   1
    #define KHRONOS_SUPPORT_FLOAT   1

    #endif


    /*
     * Types that are (so far) the same on all platforms
     */
    typedef signed   char          khronos_int8_t;
    typedef unsigned char          khronos_uint8_t;
    typedef signed   short int     khronos_int16_t;
    typedef unsigned short int     khronos_uint16_t;

    /*
     * Types that differ between LLP64 and LP64 architectures - in LLP64,
     * pointers are 64 bits, but 'long' is still 32 bits. Win64 appears
     * to be the only LLP64 architecture in current use.
     */
    #ifdef KHRONOS_USE_INTPTR_T
    typedef intptr_t               khronos_intptr_t;
    typedef uintptr_t              khronos_uintptr_t;
    #elif defined(_WIN64)
    typedef signed   long long int khronos_intptr_t;
    typedef unsigned long long int khronos_uintptr_t;
    #else
    typedef signed   long  int     khronos_intptr_t;
    typedef unsigned long  int     khronos_uintptr_t;
    #endif

    #if defined(_WIN64)
    typedef signed   long long int khronos_ssize_t;
    typedef unsigned long long int khronos_usize_t;
    #else
    typedef signed   long  int     khronos_ssize_t;
    typedef unsigned long  int     khronos_usize_t;
    #endif

    #if KHRONOS_SUPPORT_FLOAT
    /*
     * Float type
     */
    typedef          float         khronos_float_t;
    #endif

    #if KHRONOS_SUPPORT_INT64
    /* Time types
     *
     * These types can be used to represent a time interval in nanoseconds or
     * an absolute Unadjusted System Time.  Unadjusted System Time is the number
     * of nanoseconds since some arbitrary system event (e.g. since the last
     * time the system booted).  The Unadjusted System Time is an unsigned
     * 64 bit value that wraps back to 0 every 584 years.  Time intervals
     * may be either signed or unsigned.
     */
    typedef khronos_uint64_t       khronos_utime_nanoseconds_t;
    typedef khronos_int64_t        khronos_stime_nanoseconds_t;
    #endif

    /*
     * Dummy value used to pad enum types to 32 bits.
     */
    #ifndef KHRONOS_MAX_ENUM
    #define KHRONOS_MAX_ENUM 0x7FFFFFFF
    #endif

     /*
      * Enumerated boolean type
      *
      * Values other than zero should be considered to be true.  Therefore
      * comparisons should not be made against KHRONOS_TRUE.
      */
    typedef enum {
      KHRONOS_FALSE = 0,
      KHRONOS_TRUE = 1,
      KHRONOS_BOOLEAN_ENUM_FORCE_SIZE = KHRONOS_MAX_ENUM
    } khronos_boolean_enum_t;

    #endif /* __khrplatform_h_ */
    // khronos types -


    #ifdef APIENTRY
    #define FAN_API_PTR APIENTRY
    #elif defined(fan_platform_windows)
    #define FAN_API_PTR __stdcall
    #else
    #define FAN_API_PTR
    #endif

    typedef unsigned int GLenum;
    typedef unsigned char GLboolean;
    typedef unsigned int GLbitfield;
    typedef void GLvoid;
    typedef khronos_int8_t GLbyte;
    typedef khronos_uint8_t GLubyte;
    typedef khronos_int16_t GLshort;
    typedef khronos_uint16_t GLushort;
    typedef int GLint;
    typedef unsigned int GLuint;
    typedef khronos_int32_t GLclampx;
    typedef int GLsizei;
    typedef khronos_float_t GLfloat;
    typedef khronos_float_t GLclampf;
    typedef double GLdouble;
    typedef double GLclampd;
    typedef void* GLeglClientBufferEXT;
    typedef void* GLeglImageOES;
    typedef char GLchar;
    typedef char GLcharARB;

    typedef unsigned int GLenum;
    typedef unsigned char GLboolean;
    typedef unsigned int GLbitfield;
    typedef void GLvoid;
    typedef khronos_int8_t GLbyte;
    typedef khronos_uint8_t GLubyte;
    typedef khronos_int16_t GLshort;
    typedef khronos_uint16_t GLushort;
    typedef int GLint;
    typedef unsigned int GLuint;
    typedef khronos_int32_t GLclampx;
    typedef int GLsizei;
    typedef khronos_float_t GLfloat;
    typedef khronos_float_t GLclampf;
    typedef double GLdouble;
    typedef double GLclampd;
    typedef void* GLeglClientBufferEXT;
    typedef void* GLeglImageOES;
    typedef char GLchar;
    typedef char GLcharARB;
    #ifdef __APPLE__
    typedef void* GLhandleARB;
    #else
    typedef unsigned int GLhandleARB;
    #endif
    typedef khronos_uint16_t GLhalf;
    typedef khronos_uint16_t GLhalfARB;
    typedef khronos_int32_t GLfixed;
    #if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1060)
    typedef khronos_intptr_t GLintptr;
    #else
    typedef khronos_intptr_t GLintptr;
    #endif
    #if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1060)
    typedef khronos_intptr_t GLintptrARB;
    #else
    typedef khronos_intptr_t GLintptrARB;
    #endif
    #if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1060)
    typedef khronos_ssize_t GLsizeiptr;
    #else
    typedef khronos_ssize_t GLsizeiptr;
    #endif
    #if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1060)
    typedef khronos_ssize_t GLsizeiptrARB;
    #else
    typedef khronos_ssize_t GLsizeiptrARB;
    #endif
    typedef khronos_int64_t GLint64;
    typedef khronos_int64_t GLint64EXT;
    typedef khronos_uint64_t GLuint64;
    typedef khronos_uint64_t GLuint64EXT;
    typedef struct __GLsync* GLsync;
    struct _cl_context;
    struct _cl_event;
    typedef void (FAN_API_PTR* GLDEBUGPROC)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
    typedef void (FAN_API_PTR* GLDEBUGPROCARB)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
    typedef void (FAN_API_PTR* GLDEBUGPROCKHR)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
    typedef void (FAN_API_PTR* GLDEBUGPROCAMD)(GLuint id, GLenum category, GLenum severity, GLsizei length, const GLchar* message, void* userParam);
    typedef unsigned short GLhalfNV;
    typedef GLintptr GLvdpauSurfaceNV;
    typedef void (FAN_API_PTR* GLVULKANPROCNV)(void);

    typedef void (FAN_API_PTR* PFNGLACCUMPROC)(GLenum op, GLfloat value);
    typedef void (FAN_API_PTR* PFNGLACTIVESHADERPROGRAMPROC)(GLuint pipeline, GLuint program);
    typedef void (FAN_API_PTR* PFNGLACTIVETEXTUREPROC)(GLenum texture);
    typedef void (FAN_API_PTR* PFNGLALPHAFUNCPROC)(GLenum func, GLfloat ref);
    typedef GLboolean(FAN_API_PTR* PFNGLARETEXTURESRESIDENTPROC)(GLsizei n, const GLuint* textures, GLboolean* residences);
    typedef void (FAN_API_PTR* PFNGLARRAYELEMENTPROC)(GLint i);
    typedef void (FAN_API_PTR* PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
    typedef void (FAN_API_PTR* PFNGLBEGINPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLBEGINCONDITIONALRENDERPROC)(GLuint id, GLenum mode);
    typedef void (FAN_API_PTR* PFNGLBEGINQUERYPROC)(GLenum target, GLuint id);
    typedef void (FAN_API_PTR* PFNGLBEGINQUERYINDEXEDPROC)(GLenum target, GLuint index, GLuint id);
    typedef void (FAN_API_PTR* PFNGLBEGINTRANSFORMFEEDBACKPROC)(GLenum primitiveMode);
    typedef void (FAN_API_PTR* PFNGLBINDATTRIBLOCATIONPROC)(GLuint program, GLuint index, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLBINDBUFFERBASEPROC)(GLenum target, GLuint index, GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLBINDBUFFERRANGEPROC)(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
    typedef void (FAN_API_PTR* PFNGLBINDBUFFERSBASEPROC)(GLenum target, GLuint first, GLsizei count, const GLuint* buffers);
    typedef void (FAN_API_PTR* PFNGLBINDBUFFERSRANGEPROC)(GLenum target, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes);
    typedef void (FAN_API_PTR* PFNGLBINDFRAGDATALOCATIONPROC)(GLuint program, GLuint color, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLBINDFRAGDATALOCATIONINDEXEDPROC)(GLuint program, GLuint colorNumber, GLuint index, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLBINDFRAMEBUFFERPROC)(GLenum target, GLuint framebuffer);
    typedef void (FAN_API_PTR* PFNGLBINDIMAGETEXTUREPROC)(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);
    typedef void (FAN_API_PTR* PFNGLBINDIMAGETEXTURESPROC)(GLuint first, GLsizei count, const GLuint* textures);
    typedef void (FAN_API_PTR* PFNGLBINDPROGRAMPIPELINEPROC)(GLuint pipeline);
    typedef void (FAN_API_PTR* PFNGLBINDRENDERBUFFERPROC)(GLenum target, GLuint renderbuffer);
    typedef void (FAN_API_PTR* PFNGLBINDSAMPLERPROC)(GLuint unit, GLuint sampler);
    typedef void (FAN_API_PTR* PFNGLBINDSAMPLERSPROC)(GLuint first, GLsizei count, const GLuint* samplers);
    typedef void (FAN_API_PTR* PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
    typedef void (FAN_API_PTR* PFNGLBINDTEXTUREUNITPROC)(GLuint unit, GLuint texture);
    typedef void (FAN_API_PTR* PFNGLBINDTEXTURESPROC)(GLuint first, GLsizei count, const GLuint* textures);
    typedef void (FAN_API_PTR* PFNGLBINDTRANSFORMFEEDBACKPROC)(GLenum target, GLuint id);
    typedef void (FAN_API_PTR* PFNGLBINDVERTEXARRAYPROC)(GLuint array);
    typedef void (FAN_API_PTR* PFNGLBINDVERTEXBUFFERPROC)(GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride);
    typedef void (FAN_API_PTR* PFNGLBINDVERTEXBUFFERSPROC)(GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizei* strides);
    typedef void (FAN_API_PTR* PFNGLBITMAPPROC)(GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte* bitmap);
    typedef void (FAN_API_PTR* PFNGLBLENDCOLORPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
    typedef void (FAN_API_PTR* PFNGLBLENDEQUATIONPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLBLENDEQUATIONSEPARATEPROC)(GLenum modeRGB, GLenum modeAlpha);
    typedef void (FAN_API_PTR* PFNGLBLENDEQUATIONSEPARATEIPROC)(GLuint buf, GLenum modeRGB, GLenum modeAlpha);
    typedef void (FAN_API_PTR* PFNGLBLENDEQUATIONIPROC)(GLuint buf, GLenum mode);
    typedef void (FAN_API_PTR* PFNGLBLENDFUNCPROC)(GLenum sfactor, GLenum dfactor);
    typedef void (FAN_API_PTR* PFNGLBLENDFUNCSEPARATEPROC)(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
    typedef void (FAN_API_PTR* PFNGLBLENDFUNCSEPARATEIPROC)(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
    typedef void (FAN_API_PTR* PFNGLBLENDFUNCIPROC)(GLuint buf, GLenum src, GLenum dst);
    typedef void (FAN_API_PTR* PFNGLBLITFRAMEBUFFERPROC)(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum image_filter);
    typedef void (FAN_API_PTR* PFNGLBLITNAMEDFRAMEBUFFERPROC)(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum image_filter);
    typedef void (FAN_API_PTR* PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void* data, GLenum usage);
    typedef void (FAN_API_PTR* PFNGLBUFFERSTORAGEPROC)(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags);
    typedef void (FAN_API_PTR* PFNGLBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, const void* data);
    typedef void (FAN_API_PTR* PFNGLCALLLISTPROC)(GLuint list);
    typedef void (FAN_API_PTR* PFNGLCALLLISTSPROC)(GLsizei n, GLenum type, const void* lists);
    typedef GLenum(FAN_API_PTR* PFNGLCHECKFRAMEBUFFERSTATUSPROC)(GLenum target);
    typedef GLenum(FAN_API_PTR* PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC)(GLuint framebuffer, GLenum target);
    typedef void (FAN_API_PTR* PFNGLCLAMPCOLORPROC)(GLenum target, GLenum clamp);
    typedef void (FAN_API_PTR* PFNGLCLEARPROC)(GLbitfield mask);
    typedef void (FAN_API_PTR* PFNGLCLEARACCUMPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
    typedef void (FAN_API_PTR* PFNGLCLEARBUFFERDATAPROC)(GLenum target, GLenum internalformat, GLenum format, GLenum type, const void* data);
    typedef void (FAN_API_PTR* PFNGLCLEARBUFFERSUBDATAPROC)(GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data);
    typedef void (FAN_API_PTR* PFNGLCLEARBUFFERFIPROC)(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
    typedef void (FAN_API_PTR* PFNGLCLEARBUFFERFVPROC)(GLenum buffer, GLint drawbuffer, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLCLEARBUFFERIVPROC)(GLenum buffer, GLint drawbuffer, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLCLEARBUFFERUIVPROC)(GLenum buffer, GLint drawbuffer, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLCLEARCOLORPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
    typedef void (FAN_API_PTR* PFNGLCLEARDEPTHPROC)(GLdouble depth);
    typedef void (FAN_API_PTR* PFNGLCLEARDEPTHFPROC)(GLfloat d);
    typedef void (FAN_API_PTR* PFNGLCLEARINDEXPROC)(GLfloat c);
    typedef void (FAN_API_PTR* PFNGLCLEARNAMEDBUFFERDATAPROC)(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void* data);
    typedef void (FAN_API_PTR* PFNGLCLEARNAMEDBUFFERSUBDATAPROC)(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void* data);
    typedef void (FAN_API_PTR* PFNGLCLEARNAMEDFRAMEBUFFERFIPROC)(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
    typedef void (FAN_API_PTR* PFNGLCLEARNAMEDFRAMEBUFFERFVPROC)(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLCLEARNAMEDFRAMEBUFFERIVPROC)(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC)(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLCLEARSTENCILPROC)(GLint s);
    typedef void (FAN_API_PTR* PFNGLCLEARTEXIMAGEPROC)(GLuint texture, GLint level, GLenum format, GLenum type, const void* data);
    typedef void (FAN_API_PTR* PFNGLCLEARTEXSUBIMAGEPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* data);
    typedef void (FAN_API_PTR* PFNGLCLIENTACTIVETEXTUREPROC)(GLenum texture);
    typedef GLenum(FAN_API_PTR* PFNGLCLIENTWAITSYNCPROC)(GLsync sync, GLbitfield flags, GLuint64 timeout);
    typedef void (FAN_API_PTR* PFNGLCLIPCONTROLPROC)(GLenum origin, GLenum depth);
    typedef void (FAN_API_PTR* PFNGLCLIPPLANEPROC)(GLenum plane, const GLdouble* equation);
    typedef void (FAN_API_PTR* PFNGLCOLOR3BPROC)(GLbyte red, GLbyte green, GLbyte blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3BVPROC)(const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3DPROC)(GLdouble red, GLdouble green, GLdouble blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3FPROC)(GLfloat red, GLfloat green, GLfloat blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3IPROC)(GLint red, GLint green, GLint blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3SPROC)(GLshort red, GLshort green, GLshort blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3UBPROC)(GLubyte red, GLubyte green, GLubyte blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3UBVPROC)(const GLubyte* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3UIPROC)(GLuint red, GLuint green, GLuint blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3UIVPROC)(const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR3USPROC)(GLushort red, GLushort green, GLushort blue);
    typedef void (FAN_API_PTR* PFNGLCOLOR3USVPROC)(const GLushort* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4BPROC)(GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4BVPROC)(const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4DPROC)(GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4FPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4IPROC)(GLint red, GLint green, GLint blue, GLint alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4SPROC)(GLshort red, GLshort green, GLshort blue, GLshort alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4UBPROC)(GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4UBVPROC)(const GLubyte* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4UIPROC)(GLuint red, GLuint green, GLuint blue, GLuint alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4UIVPROC)(const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLCOLOR4USPROC)(GLushort red, GLushort green, GLushort blue, GLushort alpha);
    typedef void (FAN_API_PTR* PFNGLCOLOR4USVPROC)(const GLushort* v);
    typedef void (FAN_API_PTR* PFNGLCOLORMASKPROC)(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
    typedef void (FAN_API_PTR* PFNGLCOLORMASKIPROC)(GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a);
    typedef void (FAN_API_PTR* PFNGLCOLORMATERIALPROC)(GLenum face, GLenum mode);
    typedef void (FAN_API_PTR* PFNGLCOLORP3UIPROC)(GLenum type, GLuint color);
    typedef void (FAN_API_PTR* PFNGLCOLORP3UIVPROC)(GLenum type, const GLuint* color);
    typedef void (FAN_API_PTR* PFNGLCOLORP4UIPROC)(GLenum type, GLuint color);
    typedef void (FAN_API_PTR* PFNGLCOLORP4UIVPROC)(GLenum type, const GLuint* color);
    typedef void (FAN_API_PTR* PFNGLCOLORPOINTERPROC)(GLint size, GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLCOMPILESHADERPROC)(GLuint shader);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXIMAGE1DPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXIMAGE2DPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXIMAGE3DPROC)(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC)(GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data);
    typedef void (FAN_API_PTR* PFNGLCOPYBUFFERSUBDATAPROC)(GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
    typedef void (FAN_API_PTR* PFNGLCOPYIMAGESUBDATAPROC)(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth);
    typedef void (FAN_API_PTR* PFNGLCOPYNAMEDBUFFERSUBDATAPROC)(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
    typedef void (FAN_API_PTR* PFNGLCOPYPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum type);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXIMAGE1DPROC)(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXIMAGE2DPROC)(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXSUBIMAGE1DPROC)(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXSUBIMAGE2DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXTURESUBIMAGE1DPROC)(GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXTURESUBIMAGE2DPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLCOPYTEXTURESUBIMAGE3DPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLCREATEBUFFERSPROC)(GLsizei n, GLuint* buffers);
    typedef void (FAN_API_PTR* PFNGLCREATEFRAMEBUFFERSPROC)(GLsizei n, GLuint* framebuffers);
    typedef GLuint(FAN_API_PTR* PFNGLCREATEPROGRAMPROC)(void);
    typedef void (FAN_API_PTR* PFNGLCREATEPROGRAMPIPELINESPROC)(GLsizei n, GLuint* pipelines);
    typedef void (FAN_API_PTR* PFNGLCREATEQUERIESPROC)(GLenum target, GLsizei n, GLuint* ids);
    typedef void (FAN_API_PTR* PFNGLCREATERENDERBUFFERSPROC)(GLsizei n, GLuint* renderbuffers);
    typedef void (FAN_API_PTR* PFNGLCREATESAMPLERSPROC)(GLsizei n, GLuint* samplers);
    typedef GLuint(FAN_API_PTR* PFNGLCREATESHADERPROC)(GLenum type);
    typedef GLuint(FAN_API_PTR* PFNGLCREATESHADERPROGRAMVPROC)(GLenum type, GLsizei count, const GLchar* const* strings);
    typedef void (FAN_API_PTR* PFNGLCREATETEXTURESPROC)(GLenum target, GLsizei n, GLuint* textures);
    typedef void (FAN_API_PTR* PFNGLCREATETRANSFORMFEEDBACKSPROC)(GLsizei n, GLuint* ids);
    typedef void (FAN_API_PTR* PFNGLCREATEVERTEXARRAYSPROC)(GLsizei n, GLuint* arrays);
    typedef void (FAN_API_PTR* PFNGLCULLFACEPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLDEBUGMESSAGECALLBACKPROC)(GLDEBUGPROC callback, const void* userParam);
    typedef void (FAN_API_PTR* PFNGLDEBUGMESSAGECONTROLPROC)(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled);
    typedef void (FAN_API_PTR* PFNGLDEBUGMESSAGEINSERTPROC)(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* buf);
    typedef void (FAN_API_PTR* PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint* buffers);
    typedef void (FAN_API_PTR* PFNGLDELETEFRAMEBUFFERSPROC)(GLsizei n, const GLuint* framebuffers);
    typedef void (FAN_API_PTR* PFNGLDELETELISTSPROC)(GLuint list, GLsizei range);
    typedef void (FAN_API_PTR* PFNGLDELETEPROGRAMPROC)(GLuint program);
    typedef void (FAN_API_PTR* PFNGLDELETEPROGRAMPIPELINESPROC)(GLsizei n, const GLuint* pipelines);
    typedef void (FAN_API_PTR* PFNGLDELETEQUERIESPROC)(GLsizei n, const GLuint* ids);
    typedef void (FAN_API_PTR* PFNGLDELETERENDERBUFFERSPROC)(GLsizei n, const GLuint* renderbuffers);
    typedef void (FAN_API_PTR* PFNGLDELETESAMPLERSPROC)(GLsizei count, const GLuint* samplers);
    typedef void (FAN_API_PTR* PFNGLDELETESHADERPROC)(GLuint shader);
    typedef void (FAN_API_PTR* PFNGLDELETESYNCPROC)(GLsync sync);
    typedef void (FAN_API_PTR* PFNGLDELETETEXTURESPROC)(GLsizei n, const GLuint* textures);
    typedef void (FAN_API_PTR* PFNGLDELETETRANSFORMFEEDBACKSPROC)(GLsizei n, const GLuint* ids);
    typedef void (FAN_API_PTR* PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint* arrays);
    typedef void (FAN_API_PTR* PFNGLDEPTHFUNCPROC)(GLenum func);
    typedef void (FAN_API_PTR* PFNGLDEPTHMASKPROC)(GLboolean flag);
    typedef void (FAN_API_PTR* PFNGLDEPTHRANGEPROC)(GLdouble n, GLdouble f);
    typedef void (FAN_API_PTR* PFNGLDEPTHRANGEARRAYVPROC)(GLuint first, GLsizei count, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLDEPTHRANGEINDEXEDPROC)(GLuint index, GLdouble n, GLdouble f);
    typedef void (FAN_API_PTR* PFNGLDEPTHRANGEFPROC)(GLfloat n, GLfloat f);
    typedef void (FAN_API_PTR* PFNGLDETACHSHADERPROC)(GLuint program, GLuint shader);
    typedef void (FAN_API_PTR* PFNGLDISABLEPROC)(GLenum cap);
    typedef void (FAN_API_PTR* PFNGLDISABLECLIENTSTATEPROC)(GLenum array);
    typedef void (FAN_API_PTR* PFNGLDISABLEVERTEXARRAYATTRIBPROC)(GLuint vaobj, GLuint index);
    typedef void (FAN_API_PTR* PFNGLDISABLEVERTEXATTRIBARRAYPROC)(GLuint index);
    typedef void (FAN_API_PTR* PFNGLDISABLEIPROC)(GLenum target, GLuint index);
    typedef void (FAN_API_PTR* PFNGLDISPATCHCOMPUTEPROC)(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
    typedef void (FAN_API_PTR* PFNGLDISPATCHCOMPUTEINDIRECTPROC)(GLintptr indirect);
    typedef void (FAN_API_PTR* PFNGLDRAWARRAYSPROC)(GLenum mode, GLint first, GLsizei count);
    typedef void (FAN_API_PTR* PFNGLDRAWARRAYSINDIRECTPROC)(GLenum mode, const void* indirect);
    typedef void (FAN_API_PTR* PFNGLDRAWARRAYSINSTANCEDPROC)(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
    typedef void (FAN_API_PTR* PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC)(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance);
    typedef void (FAN_API_PTR* PFNGLDRAWBUFFERPROC)(GLenum buf);
    typedef void (FAN_API_PTR* PFNGLDRAWBUFFERSPROC)(GLsizei n, const GLenum* bufs);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSPROC)(GLenum mode, GLsizei count, GLenum type, const void* indices);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSBASEVERTEXPROC)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint basevertex);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSINDIRECTPROC)(GLenum mode, GLenum type, const void* indirect);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSINSTANCEDPROC)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLuint baseinstance);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex);
    typedef void (FAN_API_PTR* PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC)(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance);
    typedef void (FAN_API_PTR* PFNGLDRAWPIXELSPROC)(GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLDRAWRANGEELEMENTSPROC)(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices);
    typedef void (FAN_API_PTR* PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC)(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices, GLint basevertex);
    typedef void (FAN_API_PTR* PFNGLDRAWTRANSFORMFEEDBACKPROC)(GLenum mode, GLuint id);
    typedef void (FAN_API_PTR* PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC)(GLenum mode, GLuint id, GLsizei instancecount);
    typedef void (FAN_API_PTR* PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC)(GLenum mode, GLuint id, GLuint stream);
    typedef void (FAN_API_PTR* PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC)(GLenum mode, GLuint id, GLuint stream, GLsizei instancecount);
    typedef void (FAN_API_PTR* PFNGLEDGEFLAGPROC)(GLboolean flag);
    typedef void (FAN_API_PTR* PFNGLEDGEFLAGPOINTERPROC)(GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLEDGEFLAGVPROC)(const GLboolean* flag);
    typedef void (FAN_API_PTR* PFNGLENABLEPROC)(GLenum cap);
    typedef void (FAN_API_PTR* PFNGLENABLECLIENTSTATEPROC)(GLenum array);
    typedef void (FAN_API_PTR* PFNGLENABLEVERTEXARRAYATTRIBPROC)(GLuint vaobj, GLuint index);
    typedef void (FAN_API_PTR* PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
    typedef void (FAN_API_PTR* PFNGLENABLEIPROC)(GLenum target, GLuint index);
    typedef void (FAN_API_PTR* PFNGLENDPROC)(void);
    typedef void (FAN_API_PTR* PFNGLENDCONDITIONALRENDERPROC)(void);
    typedef void (FAN_API_PTR* PFNGLENDLISTPROC)(void);
    typedef void (FAN_API_PTR* PFNGLENDQUERYPROC)(GLenum target);
    typedef void (FAN_API_PTR* PFNGLENDQUERYINDEXEDPROC)(GLenum target, GLuint index);
    typedef void (FAN_API_PTR* PFNGLENDTRANSFORMFEEDBACKPROC)(void);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD1DPROC)(GLdouble u);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD1DVPROC)(const GLdouble* u);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD1FPROC)(GLfloat u);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD1FVPROC)(const GLfloat* u);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD2DPROC)(GLdouble u, GLdouble v);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD2DVPROC)(const GLdouble* u);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD2FPROC)(GLfloat u, GLfloat v);
    typedef void (FAN_API_PTR* PFNGLEVALCOORD2FVPROC)(const GLfloat* u);
    typedef void (FAN_API_PTR* PFNGLEVALMESH1PROC)(GLenum mode, GLint i1, GLint i2);
    typedef void (FAN_API_PTR* PFNGLEVALMESH2PROC)(GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2);
    typedef void (FAN_API_PTR* PFNGLEVALPOINT1PROC)(GLint i);
    typedef void (FAN_API_PTR* PFNGLEVALPOINT2PROC)(GLint i, GLint j);
    typedef void (FAN_API_PTR* PFNGLFEEDBACKBUFFERPROC)(GLsizei size, GLenum type, GLfloat* buffer);
    typedef GLsync(FAN_API_PTR* PFNGLFENCESYNCPROC)(GLenum condition, GLbitfield flags);
    typedef void (FAN_API_PTR* PFNGLFINISHPROC)(void);
    typedef void (FAN_API_PTR* PFNGLFLUSHPROC)(void);
    typedef void (FAN_API_PTR* PFNGLFLUSHMAPPEDBUFFERRANGEPROC)(GLenum target, GLintptr offset, GLsizeiptr length);
    typedef void (FAN_API_PTR* PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC)(GLuint buffer, GLintptr offset, GLsizeiptr length);
    typedef void (FAN_API_PTR* PFNGLFOGCOORDPOINTERPROC)(GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLFOGCOORDDPROC)(GLdouble coord);
    typedef void (FAN_API_PTR* PFNGLFOGCOORDDVPROC)(const GLdouble* coord);
    typedef void (FAN_API_PTR* PFNGLFOGCOORDFPROC)(GLfloat coord);
    typedef void (FAN_API_PTR* PFNGLFOGCOORDFVPROC)(const GLfloat* coord);
    typedef void (FAN_API_PTR* PFNGLFOGFPROC)(GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLFOGFVPROC)(GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLFOGIPROC)(GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLFOGIVPROC)(GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERRENDERBUFFERPROC)(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERTEXTUREPROC)(GLenum target, GLenum attachment, GLuint texture, GLint level);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERTEXTURE1DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERTEXTURE2DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERTEXTURE3DPROC)(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset);
    typedef void (FAN_API_PTR* PFNGLFRAMEBUFFERTEXTURELAYERPROC)(GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer);
    typedef void (FAN_API_PTR* PFNGLFRONTFACEPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLFRUSTUMPROC)(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
    typedef void (FAN_API_PTR* PFNGLGENBUFFERSPROC)(GLsizei n, GLuint* buffers);
    typedef void (FAN_API_PTR* PFNGLGENFRAMEBUFFERSPROC)(GLsizei n, GLuint* framebuffers);
    typedef GLuint(FAN_API_PTR* PFNGLGENLISTSPROC)(GLsizei range);
    typedef void (FAN_API_PTR* PFNGLGENPROGRAMPIPELINESPROC)(GLsizei n, GLuint* pipelines);
    typedef void (FAN_API_PTR* PFNGLGENQUERIESPROC)(GLsizei n, GLuint* ids);
    typedef void (FAN_API_PTR* PFNGLGENRENDERBUFFERSPROC)(GLsizei n, GLuint* renderbuffers);
    typedef void (FAN_API_PTR* PFNGLGENSAMPLERSPROC)(GLsizei count, GLuint* samplers);
    typedef void (FAN_API_PTR* PFNGLGENTEXTURESPROC)(GLsizei n, GLuint* textures);
    typedef void (FAN_API_PTR* PFNGLGENTRANSFORMFEEDBACKSPROC)(GLsizei n, GLuint* ids);
    typedef void (FAN_API_PTR* PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint* arrays);
    typedef void (FAN_API_PTR* PFNGLGENERATEMIPMAPPROC)(GLenum target);
    typedef void (FAN_API_PTR* PFNGLGENERATETEXTUREMIPMAPPROC)(GLuint texture);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC)(GLuint program, GLuint bufferIndex, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEATTRIBPROC)(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size, GLenum* type, GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETACTIVESUBROUTINENAMEPROC)(GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC)(GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC)(GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint* values);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEUNIFORMPROC)(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size, GLenum* type, GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC)(GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformBlockName);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEUNIFORMBLOCKIVPROC)(GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEUNIFORMNAMEPROC)(GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei* length, GLchar* uniformName);
    typedef void (FAN_API_PTR* PFNGLGETACTIVEUNIFORMSIVPROC)(GLuint program, GLsizei uniformCount, const GLuint* uniformIndices, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETATTACHEDSHADERSPROC)(GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders);
    typedef GLint(FAN_API_PTR* PFNGLGETATTRIBLOCATIONPROC)(GLuint program, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETBOOLEANI_VPROC)(GLenum target, GLuint index, GLboolean* data);
    typedef void (FAN_API_PTR* PFNGLGETBOOLEANVPROC)(GLenum pname, GLboolean* data);
    typedef void (FAN_API_PTR* PFNGLGETBUFFERPARAMETERI64VPROC)(GLenum target, GLenum pname, GLint64* params);
    typedef void (FAN_API_PTR* PFNGLGETBUFFERPARAMETERIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETBUFFERPOINTERVPROC)(GLenum target, GLenum pname, void** params);
    typedef void (FAN_API_PTR* PFNGLGETBUFFERSUBDATAPROC)(GLenum target, GLintptr offset, GLsizeiptr size, void* data);
    typedef void (FAN_API_PTR* PFNGLGETCLIPPLANEPROC)(GLenum plane, GLdouble* equation);
    typedef void (FAN_API_PTR* PFNGLGETCOMPRESSEDTEXIMAGEPROC)(GLenum target, GLint level, void* img);
    typedef void (FAN_API_PTR* PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC)(GLuint texture, GLint level, GLsizei bufSize, void* pixels);
    typedef void (FAN_API_PTR* PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void* pixels);
    typedef GLuint(FAN_API_PTR* PFNGLGETDEBUGMESSAGELOGPROC)(GLuint count, GLsizei bufSize, GLenum* sources, GLenum* types, GLuint* ids, GLenum* severities, GLsizei* lengths, GLchar* messageLog);
    typedef void (FAN_API_PTR* PFNGLGETDOUBLEI_VPROC)(GLenum target, GLuint index, GLdouble* data);
    typedef void (FAN_API_PTR* PFNGLGETDOUBLEVPROC)(GLenum pname, GLdouble* data);
    typedef GLenum(FAN_API_PTR* PFNGLGETERRORPROC)(void);
    typedef void (FAN_API_PTR* PFNGLGETFLOATI_VPROC)(GLenum target, GLuint index, GLfloat* data);
    typedef void (FAN_API_PTR* PFNGLGETFLOATVPROC)(GLenum pname, GLfloat* data);
    typedef GLint(FAN_API_PTR* PFNGLGETFRAGDATAINDEXPROC)(GLuint program, const GLchar* name);
    typedef GLint(FAN_API_PTR* PFNGLGETFRAGDATALOCATIONPROC)(GLuint program, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC)(GLenum target, GLenum attachment, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETFRAMEBUFFERPARAMETERIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef GLenum(FAN_API_PTR* PFNGLGETGRAPHICSRESETSTATUSPROC)(void);
    typedef void (FAN_API_PTR* PFNGLGETINTEGER64I_VPROC)(GLenum target, GLuint index, GLint64* data);
    typedef void (FAN_API_PTR* PFNGLGETINTEGER64VPROC)(GLenum pname, GLint64* data);
    typedef void (FAN_API_PTR* PFNGLGETINTEGERI_VPROC)(GLenum target, GLuint index, GLint* data);
    typedef void (FAN_API_PTR* PFNGLGETINTEGERVPROC)(GLenum pname, GLint* data);
    typedef void (FAN_API_PTR* PFNGLGETINTERNALFORMATI64VPROC)(GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint64* params);
    typedef void (FAN_API_PTR* PFNGLGETINTERNALFORMATIVPROC)(GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETLIGHTFVPROC)(GLenum light, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETLIGHTIVPROC)(GLenum light, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETMAPDVPROC)(GLenum target, GLenum query, GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLGETMAPFVPROC)(GLenum target, GLenum query, GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLGETMAPIVPROC)(GLenum target, GLenum query, GLint* v);
    typedef void (FAN_API_PTR* PFNGLGETMATERIALFVPROC)(GLenum face, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETMATERIALIVPROC)(GLenum face, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETMULTISAMPLEFVPROC)(GLenum pname, GLuint index, GLfloat* val);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDBUFFERPARAMETERI64VPROC)(GLuint buffer, GLenum pname, GLint64* params);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDBUFFERPARAMETERIVPROC)(GLuint buffer, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDBUFFERPOINTERVPROC)(GLuint buffer, GLenum pname, void** params);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDBUFFERSUBDATAPROC)(GLuint buffer, GLintptr offset, GLsizeiptr size, void* data);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC)(GLuint framebuffer, GLenum attachment, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC)(GLuint framebuffer, GLenum pname, GLint* param);
    typedef void (FAN_API_PTR* PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC)(GLuint renderbuffer, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETOBJECTLABELPROC)(GLenum identifier, GLuint name, GLsizei bufSize, GLsizei* length, GLchar* label);
    typedef void (FAN_API_PTR* PFNGLGETOBJECTPTRLABELPROC)(const void* ptr, GLsizei bufSize, GLsizei* length, GLchar* label);
    typedef void (FAN_API_PTR* PFNGLGETPIXELMAPFVPROC)(GLenum map, GLfloat* values);
    typedef void (FAN_API_PTR* PFNGLGETPIXELMAPUIVPROC)(GLenum map, GLuint* values);
    typedef void (FAN_API_PTR* PFNGLGETPIXELMAPUSVPROC)(GLenum map, GLushort* values);
    typedef void (FAN_API_PTR* PFNGLGETPOINTERVPROC)(GLenum pname, void** params);
    typedef void (FAN_API_PTR* PFNGLGETPOLYGONSTIPPLEPROC)(GLubyte* mask);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMBINARYPROC)(GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMINTERFACEIVPROC)(GLuint program, GLenum programInterface, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMPIPELINEINFOLOGPROC)(GLuint pipeline, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMPIPELINEIVPROC)(GLuint pipeline, GLenum pname, GLint* params);
    typedef GLuint(FAN_API_PTR* PFNGLGETPROGRAMRESOURCEINDEXPROC)(GLuint program, GLenum programInterface, const GLchar* name);
    typedef GLint(FAN_API_PTR* PFNGLGETPROGRAMRESOURCELOCATIONPROC)(GLuint program, GLenum programInterface, const GLchar* name);
    typedef GLint(FAN_API_PTR* PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC)(GLuint program, GLenum programInterface, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMRESOURCENAMEPROC)(GLuint program, GLenum programInterface, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMRESOURCEIVPROC)(GLuint program, GLenum programInterface, GLuint index, GLsizei propCount, const GLenum* props, GLsizei count, GLsizei* length, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMSTAGEIVPROC)(GLuint program, GLenum shadertype, GLenum pname, GLint* values);
    typedef void (FAN_API_PTR* PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETQUERYBUFFEROBJECTI64VPROC)(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
    typedef void (FAN_API_PTR* PFNGLGETQUERYBUFFEROBJECTIVPROC)(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
    typedef void (FAN_API_PTR* PFNGLGETQUERYBUFFEROBJECTUI64VPROC)(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
    typedef void (FAN_API_PTR* PFNGLGETQUERYBUFFEROBJECTUIVPROC)(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
    typedef void (FAN_API_PTR* PFNGLGETQUERYINDEXEDIVPROC)(GLenum target, GLuint index, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETQUERYOBJECTI64VPROC)(GLuint id, GLenum pname, GLint64* params);
    typedef void (FAN_API_PTR* PFNGLGETQUERYOBJECTIVPROC)(GLuint id, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETQUERYOBJECTUI64VPROC)(GLuint id, GLenum pname, GLuint64* params);
    typedef void (FAN_API_PTR* PFNGLGETQUERYOBJECTUIVPROC)(GLuint id, GLenum pname, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETQUERYIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETRENDERBUFFERPARAMETERIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETSAMPLERPARAMETERIIVPROC)(GLuint sampler, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETSAMPLERPARAMETERIUIVPROC)(GLuint sampler, GLenum pname, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETSAMPLERPARAMETERFVPROC)(GLuint sampler, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETSAMPLERPARAMETERIVPROC)(GLuint sampler, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
    typedef void (FAN_API_PTR* PFNGLGETSHADERPRECISIONFORMATPROC)(GLenum shadertype, GLenum precisiontype, GLint* range, GLint* precision);
    typedef void (FAN_API_PTR* PFNGLGETSHADERSOURCEPROC)(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source);
    typedef void (FAN_API_PTR* PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint* params);
    typedef const GLubyte* (FAN_API_PTR* PFNGLGETSTRINGPROC)(GLenum name);
    typedef const GLubyte* (FAN_API_PTR* PFNGLGETSTRINGIPROC)(GLenum name, GLuint index);
    typedef GLuint(FAN_API_PTR* PFNGLGETSUBROUTINEINDEXPROC)(GLuint program, GLenum shadertype, const GLchar* name);
    typedef GLint(FAN_API_PTR* PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC)(GLuint program, GLenum shadertype, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETSYNCIVPROC)(GLsync sync, GLenum pname, GLsizei count, GLsizei* length, GLint* values);
    typedef void (FAN_API_PTR* PFNGLGETTEXENVFVPROC)(GLenum target, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXENVIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXGENDVPROC)(GLenum coord, GLenum pname, GLdouble* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXGENFVPROC)(GLenum coord, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXGENIVPROC)(GLenum coord, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXIMAGEPROC)(GLenum target, GLint level, GLenum format, GLenum type, void* pixels);
    typedef void (FAN_API_PTR* PFNGLGETTEXLEVELPARAMETERFVPROC)(GLenum target, GLint level, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXLEVELPARAMETERIVPROC)(GLenum target, GLint level, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXPARAMETERIIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXPARAMETERIUIVPROC)(GLenum target, GLenum pname, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXPARAMETERFVPROC)(GLenum target, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXPARAMETERIVPROC)(GLenum target, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTUREIMAGEPROC)(GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels);
    typedef void (FAN_API_PTR* PFNGLGETTEXTURELEVELPARAMETERFVPROC)(GLuint texture, GLint level, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTURELEVELPARAMETERIVPROC)(GLuint texture, GLint level, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTUREPARAMETERIIVPROC)(GLuint texture, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTUREPARAMETERIUIVPROC)(GLuint texture, GLenum pname, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTUREPARAMETERFVPROC)(GLuint texture, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTUREPARAMETERIVPROC)(GLuint texture, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETTEXTURESUBIMAGEPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void* pixels);
    typedef void (FAN_API_PTR* PFNGLGETTRANSFORMFEEDBACKVARYINGPROC)(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size, GLenum* type, GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETTRANSFORMFEEDBACKI64_VPROC)(GLuint xfb, GLenum pname, GLuint index, GLint64* param);
    typedef void (FAN_API_PTR* PFNGLGETTRANSFORMFEEDBACKI_VPROC)(GLuint xfb, GLenum pname, GLuint index, GLint* param);
    typedef void (FAN_API_PTR* PFNGLGETTRANSFORMFEEDBACKIVPROC)(GLuint xfb, GLenum pname, GLint* param);
    typedef GLuint(FAN_API_PTR* PFNGLGETUNIFORMBLOCKINDEXPROC)(GLuint program, const GLchar* uniformBlockName);
    typedef void (FAN_API_PTR* PFNGLGETUNIFORMINDICESPROC)(GLuint program, GLsizei uniformCount, const GLchar* const* uniformNames, GLuint* uniformIndices);
    typedef GLint(FAN_API_PTR* PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar* name);
    typedef void (FAN_API_PTR* PFNGLGETUNIFORMSUBROUTINEUIVPROC)(GLenum shadertype, GLint location, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETUNIFORMDVPROC)(GLuint program, GLint location, GLdouble* params);
    typedef void (FAN_API_PTR* PFNGLGETUNIFORMFVPROC)(GLuint program, GLint location, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETUNIFORMIVPROC)(GLuint program, GLint location, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETUNIFORMUIVPROC)(GLuint program, GLint location, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXARRAYINDEXED64IVPROC)(GLuint vaobj, GLuint index, GLenum pname, GLint64* param);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXARRAYINDEXEDIVPROC)(GLuint vaobj, GLuint index, GLenum pname, GLint* param);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXARRAYIVPROC)(GLuint vaobj, GLenum pname, GLint* param);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBIIVPROC)(GLuint index, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBIUIVPROC)(GLuint index, GLenum pname, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBLDVPROC)(GLuint index, GLenum pname, GLdouble* params);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBPOINTERVPROC)(GLuint index, GLenum pname, void** pointer);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBDVPROC)(GLuint index, GLenum pname, GLdouble* params);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBFVPROC)(GLuint index, GLenum pname, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETVERTEXATTRIBIVPROC)(GLuint index, GLenum pname, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETNCOLORTABLEPROC)(GLenum target, GLenum format, GLenum type, GLsizei bufSize, void* table);
    typedef void (FAN_API_PTR* PFNGLGETNCOMPRESSEDTEXIMAGEPROC)(GLenum target, GLint lod, GLsizei bufSize, void* pixels);
    typedef void (FAN_API_PTR* PFNGLGETNCONVOLUTIONFILTERPROC)(GLenum target, GLenum format, GLenum type, GLsizei bufSize, void* image);
    typedef void (FAN_API_PTR* PFNGLGETNHISTOGRAMPROC)(GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void* values);
    typedef void (FAN_API_PTR* PFNGLGETNMAPDVPROC)(GLenum target, GLenum query, GLsizei bufSize, GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLGETNMAPFVPROC)(GLenum target, GLenum query, GLsizei bufSize, GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLGETNMAPIVPROC)(GLenum target, GLenum query, GLsizei bufSize, GLint* v);
    typedef void (FAN_API_PTR* PFNGLGETNMINMAXPROC)(GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void* values);
    typedef void (FAN_API_PTR* PFNGLGETNPIXELMAPFVPROC)(GLenum map, GLsizei bufSize, GLfloat* values);
    typedef void (FAN_API_PTR* PFNGLGETNPIXELMAPUIVPROC)(GLenum map, GLsizei bufSize, GLuint* values);
    typedef void (FAN_API_PTR* PFNGLGETNPIXELMAPUSVPROC)(GLenum map, GLsizei bufSize, GLushort* values);
    typedef void (FAN_API_PTR* PFNGLGETNPOLYGONSTIPPLEPROC)(GLsizei bufSize, GLubyte* pattern);
    typedef void (FAN_API_PTR* PFNGLGETNSEPARABLEFILTERPROC)(GLenum target, GLenum format, GLenum type, GLsizei rowBufSize, void* row, GLsizei columnBufSize, void* column, void* span);
    typedef void (FAN_API_PTR* PFNGLGETNTEXIMAGEPROC)(GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels);
    typedef void (FAN_API_PTR* PFNGLGETNUNIFORMDVPROC)(GLuint program, GLint location, GLsizei bufSize, GLdouble* params);
    typedef void (FAN_API_PTR* PFNGLGETNUNIFORMFVPROC)(GLuint program, GLint location, GLsizei bufSize, GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLGETNUNIFORMIVPROC)(GLuint program, GLint location, GLsizei bufSize, GLint* params);
    typedef void (FAN_API_PTR* PFNGLGETNUNIFORMUIVPROC)(GLuint program, GLint location, GLsizei bufSize, GLuint* params);
    typedef void (FAN_API_PTR* PFNGLHINTPROC)(GLenum target, GLenum mode);
    typedef void (FAN_API_PTR* PFNGLINDEXMASKPROC)(GLuint mask);
    typedef void (FAN_API_PTR* PFNGLINDEXPOINTERPROC)(GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLINDEXDPROC)(GLdouble c);
    typedef void (FAN_API_PTR* PFNGLINDEXDVPROC)(const GLdouble* c);
    typedef void (FAN_API_PTR* PFNGLINDEXFPROC)(GLfloat c);
    typedef void (FAN_API_PTR* PFNGLINDEXFVPROC)(const GLfloat* c);
    typedef void (FAN_API_PTR* PFNGLINDEXIPROC)(GLint c);
    typedef void (FAN_API_PTR* PFNGLINDEXIVPROC)(const GLint* c);
    typedef void (FAN_API_PTR* PFNGLINDEXSPROC)(GLshort c);
    typedef void (FAN_API_PTR* PFNGLINDEXSVPROC)(const GLshort* c);
    typedef void (FAN_API_PTR* PFNGLINDEXUBPROC)(GLubyte c);
    typedef void (FAN_API_PTR* PFNGLINDEXUBVPROC)(const GLubyte* c);
    typedef void (FAN_API_PTR* PFNGLINITNAMESPROC)(void);
    typedef void (FAN_API_PTR* PFNGLINTERLEAVEDARRAYSPROC)(GLenum format, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLINVALIDATEBUFFERDATAPROC)(GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLINVALIDATEBUFFERSUBDATAPROC)(GLuint buffer, GLintptr offset, GLsizeiptr length);
    typedef void (FAN_API_PTR* PFNGLINVALIDATEFRAMEBUFFERPROC)(GLenum target, GLsizei numAttachments, const GLenum* attachments);
    typedef void (FAN_API_PTR* PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC)(GLuint framebuffer, GLsizei numAttachments, const GLenum* attachments);
    typedef void (FAN_API_PTR* PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC)(GLuint framebuffer, GLsizei numAttachments, const GLenum* attachments, GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLINVALIDATESUBFRAMEBUFFERPROC)(GLenum target, GLsizei numAttachments, const GLenum* attachments, GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLINVALIDATETEXIMAGEPROC)(GLuint texture, GLint level);
    typedef void (FAN_API_PTR* PFNGLINVALIDATETEXSUBIMAGEPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth);
    typedef GLboolean(FAN_API_PTR* PFNGLISBUFFERPROC)(GLuint buffer);
    typedef GLboolean(FAN_API_PTR* PFNGLISENABLEDPROC)(GLenum cap);
    typedef GLboolean(FAN_API_PTR* PFNGLISENABLEDIPROC)(GLenum target, GLuint index);
    typedef GLboolean(FAN_API_PTR* PFNGLISFRAMEBUFFERPROC)(GLuint framebuffer);
    typedef GLboolean(FAN_API_PTR* PFNGLISLISTPROC)(GLuint list);
    typedef GLboolean(FAN_API_PTR* PFNGLISPROGRAMPROC)(GLuint program);
    typedef GLboolean(FAN_API_PTR* PFNGLISPROGRAMPIPELINEPROC)(GLuint pipeline);
    typedef GLboolean(FAN_API_PTR* PFNGLISQUERYPROC)(GLuint id);
    typedef GLboolean(FAN_API_PTR* PFNGLISRENDERBUFFERPROC)(GLuint renderbuffer);
    typedef GLboolean(FAN_API_PTR* PFNGLISSAMPLERPROC)(GLuint sampler);
    typedef GLboolean(FAN_API_PTR* PFNGLISSHADERPROC)(GLuint shader);
    typedef GLboolean(FAN_API_PTR* PFNGLISSYNCPROC)(GLsync sync);
    typedef GLboolean(FAN_API_PTR* PFNGLISTEXTUREPROC)(GLuint texture);
    typedef GLboolean(FAN_API_PTR* PFNGLISTRANSFORMFEEDBACKPROC)(GLuint id);
    typedef GLboolean(FAN_API_PTR* PFNGLISVERTEXARRAYPROC)(GLuint array);
    typedef void (FAN_API_PTR* PFNGLLIGHTMODELFPROC)(GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLLIGHTMODELFVPROC)(GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLLIGHTMODELIPROC)(GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLLIGHTMODELIVPROC)(GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLLIGHTFPROC)(GLenum light, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLLIGHTFVPROC)(GLenum light, GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLLIGHTIPROC)(GLenum light, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLLIGHTIVPROC)(GLenum light, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLLINESTIPPLEPROC)(GLint factor, GLushort pattern);
    typedef void (FAN_API_PTR* PFNGLLINEWIDTHPROC)(GLfloat width);
    typedef void (FAN_API_PTR* PFNGLLINKPROGRAMPROC)(GLuint program);
    typedef void (FAN_API_PTR* PFNGLLISTBASEPROC)(GLuint base);
    typedef void (FAN_API_PTR* PFNGLLOADIDENTITYPROC)(void);
    typedef void (FAN_API_PTR* PFNGLLOADMATRIXDPROC)(const GLdouble* m);
    typedef void (FAN_API_PTR* PFNGLLOADMATRIXFPROC)(const GLfloat* m);
    typedef void (FAN_API_PTR* PFNGLLOADNAMEPROC)(GLuint name);
    typedef void (FAN_API_PTR* PFNGLLOADTRANSPOSEMATRIXDPROC)(const GLdouble* m);
    typedef void (FAN_API_PTR* PFNGLLOADTRANSPOSEMATRIXFPROC)(const GLfloat* m);
    typedef void (FAN_API_PTR* PFNGLLOGICOPPROC)(GLenum opcode);
    typedef void (FAN_API_PTR* PFNGLMAP1DPROC)(GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble* points);
    typedef void (FAN_API_PTR* PFNGLMAP1FPROC)(GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat* points);
    typedef void (FAN_API_PTR* PFNGLMAP2DPROC)(GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble* points);
    typedef void (FAN_API_PTR* PFNGLMAP2FPROC)(GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder, GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat* points);
    typedef void* (FAN_API_PTR* PFNGLMAPBUFFERPROC)(GLenum target, GLenum access);
    typedef void* (FAN_API_PTR* PFNGLMAPBUFFERRANGEPROC)(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
    typedef void (FAN_API_PTR* PFNGLMAPGRID1DPROC)(GLint un, GLdouble u1, GLdouble u2);
    typedef void (FAN_API_PTR* PFNGLMAPGRID1FPROC)(GLint un, GLfloat u1, GLfloat u2);
    typedef void (FAN_API_PTR* PFNGLMAPGRID2DPROC)(GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2);
    typedef void (FAN_API_PTR* PFNGLMAPGRID2FPROC)(GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2);
    typedef void* (FAN_API_PTR* PFNGLMAPNAMEDBUFFERPROC)(GLuint buffer, GLenum access);
    typedef void* (FAN_API_PTR* PFNGLMAPNAMEDBUFFERRANGEPROC)(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access);
    typedef void (FAN_API_PTR* PFNGLMATERIALFPROC)(GLenum face, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLMATERIALFVPROC)(GLenum face, GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLMATERIALIPROC)(GLenum face, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLMATERIALIVPROC)(GLenum face, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLMATRIXMODEPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLMEMORYBARRIERPROC)(GLbitfield barriers);
    typedef void (FAN_API_PTR* PFNGLMEMORYBARRIERBYREGIONPROC)(GLbitfield barriers);
    typedef void (FAN_API_PTR* PFNGLMINSAMPLESHADINGPROC)(GLfloat value);
    typedef void (FAN_API_PTR* PFNGLMULTMATRIXDPROC)(const GLdouble* m);
    typedef void (FAN_API_PTR* PFNGLMULTMATRIXFPROC)(const GLfloat* m);
    typedef void (FAN_API_PTR* PFNGLMULTTRANSPOSEMATRIXDPROC)(const GLdouble* m);
    typedef void (FAN_API_PTR* PFNGLMULTTRANSPOSEMATRIXFPROC)(const GLfloat* m);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWARRAYSPROC)(GLenum mode, const GLint* first, const GLsizei* count, GLsizei drawcount);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWARRAYSINDIRECTPROC)(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWARRAYSINDIRECTCOUNTPROC)(GLenum mode, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWELEMENTSPROC)(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC)(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount, const GLint* basevertex);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWELEMENTSINDIRECTPROC)(GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride);
    typedef void (FAN_API_PTR* PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTPROC)(GLenum mode, GLenum type, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1DPROC)(GLenum target, GLdouble s);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1DVPROC)(GLenum target, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1FPROC)(GLenum target, GLfloat s);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1FVPROC)(GLenum target, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1IPROC)(GLenum target, GLint s);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1IVPROC)(GLenum target, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1SPROC)(GLenum target, GLshort s);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD1SVPROC)(GLenum target, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2DPROC)(GLenum target, GLdouble s, GLdouble t);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2DVPROC)(GLenum target, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2FPROC)(GLenum target, GLfloat s, GLfloat t);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2FVPROC)(GLenum target, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2IPROC)(GLenum target, GLint s, GLint t);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2IVPROC)(GLenum target, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2SPROC)(GLenum target, GLshort s, GLshort t);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD2SVPROC)(GLenum target, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3DPROC)(GLenum target, GLdouble s, GLdouble t, GLdouble r);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3DVPROC)(GLenum target, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3FPROC)(GLenum target, GLfloat s, GLfloat t, GLfloat r);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3FVPROC)(GLenum target, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3IPROC)(GLenum target, GLint s, GLint t, GLint r);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3IVPROC)(GLenum target, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3SPROC)(GLenum target, GLshort s, GLshort t, GLshort r);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD3SVPROC)(GLenum target, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4DPROC)(GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4DVPROC)(GLenum target, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4FPROC)(GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4FVPROC)(GLenum target, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4IPROC)(GLenum target, GLint s, GLint t, GLint r, GLint q);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4IVPROC)(GLenum target, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4SPROC)(GLenum target, GLshort s, GLshort t, GLshort r, GLshort q);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORD4SVPROC)(GLenum target, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP1UIPROC)(GLenum texture, GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP1UIVPROC)(GLenum texture, GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP2UIPROC)(GLenum texture, GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP2UIVPROC)(GLenum texture, GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP3UIPROC)(GLenum texture, GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP3UIVPROC)(GLenum texture, GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP4UIPROC)(GLenum texture, GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLMULTITEXCOORDP4UIVPROC)(GLenum texture, GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLNAMEDBUFFERDATAPROC)(GLuint buffer, GLsizeiptr size, const void* data, GLenum usage);
    typedef void (FAN_API_PTR* PFNGLNAMEDBUFFERSTORAGEPROC)(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags);
    typedef void (FAN_API_PTR* PFNGLNAMEDBUFFERSUBDATAPROC)(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC)(GLuint framebuffer, GLenum buf);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC)(GLuint framebuffer, GLsizei n, const GLenum* bufs);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC)(GLuint framebuffer, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC)(GLuint framebuffer, GLenum src);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC)(GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERTEXTUREPROC)(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level);
    typedef void (FAN_API_PTR* PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC)(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer);
    typedef void (FAN_API_PTR* PFNGLNAMEDRENDERBUFFERSTORAGEPROC)(GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC)(GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLNEWLISTPROC)(GLuint list, GLenum mode);
    typedef void (FAN_API_PTR* PFNGLNORMAL3BPROC)(GLbyte nx, GLbyte ny, GLbyte nz);
    typedef void (FAN_API_PTR* PFNGLNORMAL3BVPROC)(const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLNORMAL3DPROC)(GLdouble nx, GLdouble ny, GLdouble nz);
    typedef void (FAN_API_PTR* PFNGLNORMAL3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLNORMAL3FPROC)(GLfloat nx, GLfloat ny, GLfloat nz);
    typedef void (FAN_API_PTR* PFNGLNORMAL3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLNORMAL3IPROC)(GLint nx, GLint ny, GLint nz);
    typedef void (FAN_API_PTR* PFNGLNORMAL3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLNORMAL3SPROC)(GLshort nx, GLshort ny, GLshort nz);
    typedef void (FAN_API_PTR* PFNGLNORMAL3SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLNORMALP3UIPROC)(GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLNORMALP3UIVPROC)(GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLNORMALPOINTERPROC)(GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLOBJECTLABELPROC)(GLenum identifier, GLuint name, GLsizei length, const GLchar* label);
    typedef void (FAN_API_PTR* PFNGLOBJECTPTRLABELPROC)(const void* ptr, GLsizei length, const GLchar* label);
    typedef void (FAN_API_PTR* PFNGLORTHOPROC)(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar);
    typedef void (FAN_API_PTR* PFNGLPASSTHROUGHPROC)(GLfloat token);
    typedef void (FAN_API_PTR* PFNGLPATCHPARAMETERFVPROC)(GLenum pname, const GLfloat* values);
    typedef void (FAN_API_PTR* PFNGLPATCHPARAMETERIPROC)(GLenum pname, GLint value);
    typedef void (FAN_API_PTR* PFNGLPAUSETRANSFORMFEEDBACKPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPIXELMAPFVPROC)(GLenum map, GLsizei mapsize, const GLfloat* values);
    typedef void (FAN_API_PTR* PFNGLPIXELMAPUIVPROC)(GLenum map, GLsizei mapsize, const GLuint* values);
    typedef void (FAN_API_PTR* PFNGLPIXELMAPUSVPROC)(GLenum map, GLsizei mapsize, const GLushort* values);
    typedef void (FAN_API_PTR* PFNGLPIXELSTOREFPROC)(GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLPIXELSTOREIPROC)(GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLPIXELTRANSFERFPROC)(GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLPIXELTRANSFERIPROC)(GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLPIXELZOOMPROC)(GLfloat xfactor, GLfloat yfactor);
    typedef void (FAN_API_PTR* PFNGLPOINTPARAMETERFPROC)(GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLPOINTPARAMETERFVPROC)(GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLPOINTPARAMETERIPROC)(GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLPOINTPARAMETERIVPROC)(GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLPOINTSIZEPROC)(GLfloat size);
    typedef void (FAN_API_PTR* PFNGLPOLYGONMODEPROC)(GLenum face, GLenum mode);
    typedef void (FAN_API_PTR* PFNGLPOLYGONOFFSETPROC)(GLfloat factor, GLfloat units);
    typedef void (FAN_API_PTR* PFNGLPOLYGONOFFSETCLAMPPROC)(GLfloat factor, GLfloat units, GLfloat clamp);
    typedef void (FAN_API_PTR* PFNGLPOLYGONSTIPPLEPROC)(const GLubyte* mask);
    typedef void (FAN_API_PTR* PFNGLPOPATTRIBPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPOPCLIENTATTRIBPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPOPDEBUGGROUPPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPOPMATRIXPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPOPNAMEPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPRIMITIVERESTARTINDEXPROC)(GLuint index);
    typedef void (FAN_API_PTR* PFNGLPRIORITIZETEXTURESPROC)(GLsizei n, const GLuint* textures, const GLfloat* priorities);
    typedef void (FAN_API_PTR* PFNGLPROGRAMBINARYPROC)(GLuint program, GLenum binaryFormat, const void* binary, GLsizei length);
    typedef void (FAN_API_PTR* PFNGLPROGRAMPARAMETERIPROC)(GLuint program, GLenum pname, GLint value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1DPROC)(GLuint program, GLint location, GLdouble v0);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1DVPROC)(GLuint program, GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1FPROC)(GLuint program, GLint location, GLfloat v0);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1FVPROC)(GLuint program, GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1IPROC)(GLuint program, GLint location, GLint v0);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1IVPROC)(GLuint program, GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1UIPROC)(GLuint program, GLint location, GLuint v0);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM1UIVPROC)(GLuint program, GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2DPROC)(GLuint program, GLint location, GLdouble v0, GLdouble v1);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2DVPROC)(GLuint program, GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2FPROC)(GLuint program, GLint location, GLfloat v0, GLfloat v1);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2FVPROC)(GLuint program, GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2IPROC)(GLuint program, GLint location, GLint v0, GLint v1);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2IVPROC)(GLuint program, GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2UIPROC)(GLuint program, GLint location, GLuint v0, GLuint v1);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM2UIVPROC)(GLuint program, GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3DPROC)(GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3DVPROC)(GLuint program, GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3FPROC)(GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3FVPROC)(GLuint program, GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3IPROC)(GLuint program, GLint location, GLint v0, GLint v1, GLint v2);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3IVPROC)(GLuint program, GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3UIPROC)(GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM3UIVPROC)(GLuint program, GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4DPROC)(GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4DVPROC)(GLuint program, GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4FPROC)(GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4FVPROC)(GLuint program, GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4IPROC)(GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4IVPROC)(GLuint program, GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4UIPROC)(GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORM4UIVPROC)(GLuint program, GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX2DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX2FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX3DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX3FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX4DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX4FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC)(GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLPROVOKINGVERTEXPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLPUSHATTRIBPROC)(GLbitfield mask);
    typedef void (FAN_API_PTR* PFNGLPUSHCLIENTATTRIBPROC)(GLbitfield mask);
    typedef void (FAN_API_PTR* PFNGLPUSHDEBUGGROUPPROC)(GLenum source, GLuint id, GLsizei length, const GLchar* message);
    typedef void (FAN_API_PTR* PFNGLPUSHMATRIXPROC)(void);
    typedef void (FAN_API_PTR* PFNGLPUSHNAMEPROC)(GLuint name);
    typedef void (FAN_API_PTR* PFNGLQUERYCOUNTERPROC)(GLuint id, GLenum target);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2DPROC)(GLdouble x, GLdouble y);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2FPROC)(GLfloat x, GLfloat y);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2IPROC)(GLint x, GLint y);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2SPROC)(GLshort x, GLshort y);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS2SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3DPROC)(GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3FPROC)(GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3IPROC)(GLint x, GLint y, GLint z);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3SPROC)(GLshort x, GLshort y, GLshort z);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS3SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4DPROC)(GLdouble x, GLdouble y, GLdouble z, GLdouble w);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4FPROC)(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4IPROC)(GLint x, GLint y, GLint z, GLint w);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4SPROC)(GLshort x, GLshort y, GLshort z, GLshort w);
    typedef void (FAN_API_PTR* PFNGLRASTERPOS4SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLREADBUFFERPROC)(GLenum src);
    typedef void (FAN_API_PTR* PFNGLREADPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void* pixels);
    typedef void (FAN_API_PTR* PFNGLREADNPIXELSPROC)(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void* data);
    typedef void (FAN_API_PTR* PFNGLRECTDPROC)(GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2);
    typedef void (FAN_API_PTR* PFNGLRECTDVPROC)(const GLdouble* v1, const GLdouble* v2);
    typedef void (FAN_API_PTR* PFNGLRECTFPROC)(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2);
    typedef void (FAN_API_PTR* PFNGLRECTFVPROC)(const GLfloat* v1, const GLfloat* v2);
    typedef void (FAN_API_PTR* PFNGLRECTIPROC)(GLint x1, GLint y1, GLint x2, GLint y2);
    typedef void (FAN_API_PTR* PFNGLRECTIVPROC)(const GLint* v1, const GLint* v2);
    typedef void (FAN_API_PTR* PFNGLRECTSPROC)(GLshort x1, GLshort y1, GLshort x2, GLshort y2);
    typedef void (FAN_API_PTR* PFNGLRECTSVPROC)(const GLshort* v1, const GLshort* v2);
    typedef void (FAN_API_PTR* PFNGLRELEASESHADERCOMPILERPROC)(void);
    typedef GLint(FAN_API_PTR* PFNGLRENDERMODEPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLRENDERBUFFERSTORAGEPROC)(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLRESUMETRANSFORMFEEDBACKPROC)(void);
    typedef void (FAN_API_PTR* PFNGLROTATEDPROC)(GLdouble angle, GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLROTATEFPROC)(GLfloat angle, GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLSAMPLECOVERAGEPROC)(GLfloat value, GLboolean invert);
    typedef void (FAN_API_PTR* PFNGLSAMPLEMASKIPROC)(GLuint maskNumber, GLbitfield mask);
    typedef void (FAN_API_PTR* PFNGLSAMPLERPARAMETERIIVPROC)(GLuint sampler, GLenum pname, const GLint* param);
    typedef void (FAN_API_PTR* PFNGLSAMPLERPARAMETERIUIVPROC)(GLuint sampler, GLenum pname, const GLuint* param);
    typedef void (FAN_API_PTR* PFNGLSAMPLERPARAMETERFPROC)(GLuint sampler, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLSAMPLERPARAMETERFVPROC)(GLuint sampler, GLenum pname, const GLfloat* param);
    typedef void (FAN_API_PTR* PFNGLSAMPLERPARAMETERIPROC)(GLuint sampler, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLSAMPLERPARAMETERIVPROC)(GLuint sampler, GLenum pname, const GLint* param);
    typedef void (FAN_API_PTR* PFNGLSCALEDPROC)(GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLSCALEFPROC)(GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLSCISSORPROC)(GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLSCISSORARRAYVPROC)(GLuint first, GLsizei count, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLSCISSORINDEXEDPROC)(GLuint index, GLint left, GLint bottom, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLSCISSORINDEXEDVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3BPROC)(GLbyte red, GLbyte green, GLbyte blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3BVPROC)(const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3DPROC)(GLdouble red, GLdouble green, GLdouble blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3FPROC)(GLfloat red, GLfloat green, GLfloat blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3IPROC)(GLint red, GLint green, GLint blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3SPROC)(GLshort red, GLshort green, GLshort blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3UBPROC)(GLubyte red, GLubyte green, GLubyte blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3UBVPROC)(const GLubyte* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3UIPROC)(GLuint red, GLuint green, GLuint blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3UIVPROC)(const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3USPROC)(GLushort red, GLushort green, GLushort blue);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLOR3USVPROC)(const GLushort* v);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLORP3UIPROC)(GLenum type, GLuint color);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLORP3UIVPROC)(GLenum type, const GLuint* color);
    typedef void (FAN_API_PTR* PFNGLSECONDARYCOLORPOINTERPROC)(GLint size, GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLSELECTBUFFERPROC)(GLsizei size, GLuint* buffer);
    typedef void (FAN_API_PTR* PFNGLSHADEMODELPROC)(GLenum mode);
    typedef void (FAN_API_PTR* PFNGLSHADERBINARYPROC)(GLsizei count, const GLuint* shaders, GLenum binaryFormat, const void* binary, GLsizei length);
    typedef void (FAN_API_PTR* PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
    typedef void (FAN_API_PTR* PFNGLSHADERSTORAGEBLOCKBINDINGPROC)(GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding);
    typedef void (FAN_API_PTR* PFNGLSPECIALIZESHADERPROC)(GLuint shader, const GLchar* pEntryPoint, GLuint numSpecializationConstants, const GLuint* pConstantIndex, const GLuint* pConstantValue);
    typedef void (FAN_API_PTR* PFNGLSTENCILFUNCPROC)(GLenum func, GLint ref, GLuint mask);
    typedef void (FAN_API_PTR* PFNGLSTENCILFUNCSEPARATEPROC)(GLenum face, GLenum func, GLint ref, GLuint mask);
    typedef void (FAN_API_PTR* PFNGLSTENCILMASKPROC)(GLuint mask);
    typedef void (FAN_API_PTR* PFNGLSTENCILMASKSEPARATEPROC)(GLenum face, GLuint mask);
    typedef void (FAN_API_PTR* PFNGLSTENCILOPPROC)(GLenum fail, GLenum zfail, GLenum zpass);
    typedef void (FAN_API_PTR* PFNGLSTENCILOPSEPARATEPROC)(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass);
    typedef void (FAN_API_PTR* PFNGLTEXBUFFERPROC)(GLenum target, GLenum internalformat, GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLTEXBUFFERRANGEPROC)(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1DPROC)(GLdouble s);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1FPROC)(GLfloat s);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1IPROC)(GLint s);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1SPROC)(GLshort s);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD1SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2DPROC)(GLdouble s, GLdouble t);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2FPROC)(GLfloat s, GLfloat t);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2IPROC)(GLint s, GLint t);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2SPROC)(GLshort s, GLshort t);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD2SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3DPROC)(GLdouble s, GLdouble t, GLdouble r);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3FPROC)(GLfloat s, GLfloat t, GLfloat r);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3IPROC)(GLint s, GLint t, GLint r);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3SPROC)(GLshort s, GLshort t, GLshort r);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD3SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4DPROC)(GLdouble s, GLdouble t, GLdouble r, GLdouble q);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4FPROC)(GLfloat s, GLfloat t, GLfloat r, GLfloat q);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4IPROC)(GLint s, GLint t, GLint r, GLint q);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4SPROC)(GLshort s, GLshort t, GLshort r, GLshort q);
    typedef void (FAN_API_PTR* PFNGLTEXCOORD4SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP1UIPROC)(GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP1UIVPROC)(GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP2UIPROC)(GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP2UIVPROC)(GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP3UIPROC)(GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP3UIVPROC)(GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP4UIPROC)(GLenum type, GLuint coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDP4UIVPROC)(GLenum type, const GLuint* coords);
    typedef void (FAN_API_PTR* PFNGLTEXCOORDPOINTERPROC)(GLint size, GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLTEXENVFPROC)(GLenum target, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLTEXENVFVPROC)(GLenum target, GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLTEXENVIPROC)(GLenum target, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLTEXENVIVPROC)(GLenum target, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLTEXGENDPROC)(GLenum coord, GLenum pname, GLdouble param);
    typedef void (FAN_API_PTR* PFNGLTEXGENDVPROC)(GLenum coord, GLenum pname, const GLdouble* params);
    typedef void (FAN_API_PTR* PFNGLTEXGENFPROC)(GLenum coord, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLTEXGENFVPROC)(GLenum coord, GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLTEXGENIPROC)(GLenum coord, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLTEXGENIVPROC)(GLenum coord, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLTEXIMAGE1DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXIMAGE2DMULTISAMPLEPROC)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
    typedef void (FAN_API_PTR* PFNGLTEXIMAGE3DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXIMAGE3DMULTISAMPLEPROC)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
    typedef void (FAN_API_PTR* PFNGLTEXPARAMETERIIVPROC)(GLenum target, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLTEXPARAMETERIUIVPROC)(GLenum target, GLenum pname, const GLuint* params);
    typedef void (FAN_API_PTR* PFNGLTEXPARAMETERFPROC)(GLenum target, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLTEXPARAMETERFVPROC)(GLenum target, GLenum pname, const GLfloat* params);
    typedef void (FAN_API_PTR* PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLTEXPARAMETERIVPROC)(GLenum target, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLTEXSTORAGE1DPROC)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
    typedef void (FAN_API_PTR* PFNGLTEXSTORAGE2DPROC)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLTEXSTORAGE2DMULTISAMPLEPROC)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
    typedef void (FAN_API_PTR* PFNGLTEXSTORAGE3DPROC)(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
    typedef void (FAN_API_PTR* PFNGLTEXSTORAGE3DMULTISAMPLEPROC)(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
    typedef void (FAN_API_PTR* PFNGLTEXSUBIMAGE1DPROC)(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXSUBIMAGE2DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXSUBIMAGE3DPROC)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXTUREBARRIERPROC)(void);
    typedef void (FAN_API_PTR* PFNGLTEXTUREBUFFERPROC)(GLuint texture, GLenum internalformat, GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLTEXTUREBUFFERRANGEPROC)(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size);
    typedef void (FAN_API_PTR* PFNGLTEXTUREPARAMETERIIVPROC)(GLuint texture, GLenum pname, const GLint* params);
    typedef void (FAN_API_PTR* PFNGLTEXTUREPARAMETERIUIVPROC)(GLuint texture, GLenum pname, const GLuint* params);
    typedef void (FAN_API_PTR* PFNGLTEXTUREPARAMETERFPROC)(GLuint texture, GLenum pname, GLfloat param);
    typedef void (FAN_API_PTR* PFNGLTEXTUREPARAMETERFVPROC)(GLuint texture, GLenum pname, const GLfloat* param);
    typedef void (FAN_API_PTR* PFNGLTEXTUREPARAMETERIPROC)(GLuint texture, GLenum pname, GLint param);
    typedef void (FAN_API_PTR* PFNGLTEXTUREPARAMETERIVPROC)(GLuint texture, GLenum pname, const GLint* param);
    typedef void (FAN_API_PTR* PFNGLTEXTURESTORAGE1DPROC)(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width);
    typedef void (FAN_API_PTR* PFNGLTEXTURESTORAGE2DPROC)(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC)(GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
    typedef void (FAN_API_PTR* PFNGLTEXTURESTORAGE3DPROC)(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
    typedef void (FAN_API_PTR* PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC)(GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
    typedef void (FAN_API_PTR* PFNGLTEXTURESUBIMAGE1DPROC)(GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXTURESUBIMAGE2DPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXTURESUBIMAGE3DPROC)(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels);
    typedef void (FAN_API_PTR* PFNGLTEXTUREVIEWPROC)(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers);
    typedef void (FAN_API_PTR* PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC)(GLuint xfb, GLuint index, GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC)(GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);
    typedef void (FAN_API_PTR* PFNGLTRANSFORMFEEDBACKVARYINGSPROC)(GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode);
    typedef void (FAN_API_PTR* PFNGLTRANSLATEDPROC)(GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLTRANSLATEFPROC)(GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1DPROC)(GLint location, GLdouble x);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1DVPROC)(GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1FVPROC)(GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1IVPROC)(GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1UIPROC)(GLint location, GLuint v0);
    typedef void (FAN_API_PTR* PFNGLUNIFORM1UIVPROC)(GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2DPROC)(GLint location, GLdouble x, GLdouble y);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2DVPROC)(GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2FPROC)(GLint location, GLfloat v0, GLfloat v1);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2FVPROC)(GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2IPROC)(GLint location, GLint v0, GLint v1);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2IVPROC)(GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2UIPROC)(GLint location, GLuint v0, GLuint v1);
    typedef void (FAN_API_PTR* PFNGLUNIFORM2UIVPROC)(GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3DPROC)(GLint location, GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3DVPROC)(GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3FPROC)(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3FVPROC)(GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3IPROC)(GLint location, GLint v0, GLint v1, GLint v2);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3IVPROC)(GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3UIPROC)(GLint location, GLuint v0, GLuint v1, GLuint v2);
    typedef void (FAN_API_PTR* PFNGLUNIFORM3UIVPROC)(GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4DPROC)(GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4DVPROC)(GLint location, GLsizei count, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4FPROC)(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4FVPROC)(GLint location, GLsizei count, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4IPROC)(GLint location, GLint v0, GLint v1, GLint v2, GLint v3);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4IVPROC)(GLint location, GLsizei count, const GLint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4UIPROC)(GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
    typedef void (FAN_API_PTR* PFNGLUNIFORM4UIVPROC)(GLint location, GLsizei count, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMBLOCKBINDINGPROC)(GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX2DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX2X3DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX2X3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX2X4DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX2X4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX3DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX3X2DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX3X2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX3X4DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX3X4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX4DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX4FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX4X2DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX4X2FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX4X3DVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLdouble* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMMATRIX4X3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value);
    typedef void (FAN_API_PTR* PFNGLUNIFORMSUBROUTINESUIVPROC)(GLenum shadertype, GLsizei count, const GLuint* indices);
    typedef GLboolean(FAN_API_PTR* PFNGLUNMAPBUFFERPROC)(GLenum target);
    typedef GLboolean(FAN_API_PTR* PFNGLUNMAPNAMEDBUFFERPROC)(GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLUSEPROGRAMPROC)(GLuint program);
    typedef void (FAN_API_PTR* PFNGLUSEPROGRAMSTAGESPROC)(GLuint pipeline, GLbitfield stages, GLuint program);
    typedef void (FAN_API_PTR* PFNGLVALIDATEPROGRAMPROC)(GLuint program);
    typedef void (FAN_API_PTR* PFNGLVALIDATEPROGRAMPIPELINEPROC)(GLuint pipeline);
    typedef void (FAN_API_PTR* PFNGLVERTEX2DPROC)(GLdouble x, GLdouble y);
    typedef void (FAN_API_PTR* PFNGLVERTEX2DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX2FPROC)(GLfloat x, GLfloat y);
    typedef void (FAN_API_PTR* PFNGLVERTEX2FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX2IPROC)(GLint x, GLint y);
    typedef void (FAN_API_PTR* PFNGLVERTEX2IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX2SPROC)(GLshort x, GLshort y);
    typedef void (FAN_API_PTR* PFNGLVERTEX2SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX3DPROC)(GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLVERTEX3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX3FPROC)(GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLVERTEX3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX3IPROC)(GLint x, GLint y, GLint z);
    typedef void (FAN_API_PTR* PFNGLVERTEX3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX3SPROC)(GLshort x, GLshort y, GLshort z);
    typedef void (FAN_API_PTR* PFNGLVERTEX3SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX4DPROC)(GLdouble x, GLdouble y, GLdouble z, GLdouble w);
    typedef void (FAN_API_PTR* PFNGLVERTEX4DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX4FPROC)(GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    typedef void (FAN_API_PTR* PFNGLVERTEX4FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX4IPROC)(GLint x, GLint y, GLint z, GLint w);
    typedef void (FAN_API_PTR* PFNGLVERTEX4IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEX4SPROC)(GLshort x, GLshort y, GLshort z, GLshort w);
    typedef void (FAN_API_PTR* PFNGLVERTEX4SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYATTRIBBINDINGPROC)(GLuint vaobj, GLuint attribindex, GLuint bindingindex);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYATTRIBFORMATPROC)(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYATTRIBIFORMATPROC)(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYATTRIBLFORMATPROC)(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYBINDINGDIVISORPROC)(GLuint vaobj, GLuint bindingindex, GLuint divisor);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYELEMENTBUFFERPROC)(GLuint vaobj, GLuint buffer);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYVERTEXBUFFERPROC)(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride);
    typedef void (FAN_API_PTR* PFNGLVERTEXARRAYVERTEXBUFFERSPROC)(GLuint vaobj, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizei* strides);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB1DPROC)(GLuint index, GLdouble x);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB1DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB1FPROC)(GLuint index, GLfloat x);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB1FVPROC)(GLuint index, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB1SPROC)(GLuint index, GLshort x);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB1SVPROC)(GLuint index, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB2DPROC)(GLuint index, GLdouble x, GLdouble y);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB2DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB2FPROC)(GLuint index, GLfloat x, GLfloat y);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB2FVPROC)(GLuint index, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB2SPROC)(GLuint index, GLshort x, GLshort y);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB2SVPROC)(GLuint index, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB3DPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB3DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB3FPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB3FVPROC)(GLuint index, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB3SPROC)(GLuint index, GLshort x, GLshort y, GLshort z);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB3SVPROC)(GLuint index, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NBVPROC)(GLuint index, const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NIVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NSVPROC)(GLuint index, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NUBPROC)(GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NUBVPROC)(GLuint index, const GLubyte* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NUIVPROC)(GLuint index, const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4NUSVPROC)(GLuint index, const GLushort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4BVPROC)(GLuint index, const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4DPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4FPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4FVPROC)(GLuint index, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4IVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4SPROC)(GLuint index, GLshort x, GLshort y, GLshort z, GLshort w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4SVPROC)(GLuint index, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4UBVPROC)(GLuint index, const GLubyte* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4UIVPROC)(GLuint index, const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIB4USVPROC)(GLuint index, const GLushort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBBINDINGPROC)(GLuint attribindex, GLuint bindingindex);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBDIVISORPROC)(GLuint index, GLuint divisor);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBFORMATPROC)(GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI1IPROC)(GLuint index, GLint x);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI1IVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI1UIPROC)(GLuint index, GLuint x);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI1UIVPROC)(GLuint index, const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI2IPROC)(GLuint index, GLint x, GLint y);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI2IVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI2UIPROC)(GLuint index, GLuint x, GLuint y);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI2UIVPROC)(GLuint index, const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI3IPROC)(GLuint index, GLint x, GLint y, GLint z);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI3IVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI3UIPROC)(GLuint index, GLuint x, GLuint y, GLuint z);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI3UIVPROC)(GLuint index, const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4BVPROC)(GLuint index, const GLbyte* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4IPROC)(GLuint index, GLint x, GLint y, GLint z, GLint w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4IVPROC)(GLuint index, const GLint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4SVPROC)(GLuint index, const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4UBVPROC)(GLuint index, const GLubyte* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4UIPROC)(GLuint index, GLuint x, GLuint y, GLuint z, GLuint w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4UIVPROC)(GLuint index, const GLuint* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBI4USVPROC)(GLuint index, const GLushort* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBIFORMATPROC)(GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBIPOINTERPROC)(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL1DPROC)(GLuint index, GLdouble x);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL1DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL2DPROC)(GLuint index, GLdouble x, GLdouble y);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL2DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL3DPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL3DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL4DPROC)(GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBL4DVPROC)(GLuint index, const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBLFORMATPROC)(GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBLPOINTERPROC)(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP1UIPROC)(GLuint index, GLenum type, GLboolean normalized, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP1UIVPROC)(GLuint index, GLenum type, GLboolean normalized, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP2UIPROC)(GLuint index, GLenum type, GLboolean normalized, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP2UIVPROC)(GLuint index, GLenum type, GLboolean normalized, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP3UIPROC)(GLuint index, GLenum type, GLboolean normalized, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP3UIVPROC)(GLuint index, GLenum type, GLboolean normalized, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP4UIPROC)(GLuint index, GLenum type, GLboolean normalized, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBP4UIVPROC)(GLuint index, GLenum type, GLboolean normalized, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLVERTEXBINDINGDIVISORPROC)(GLuint bindingindex, GLuint divisor);
    typedef void (FAN_API_PTR* PFNGLVERTEXP2UIPROC)(GLenum type, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXP2UIVPROC)(GLenum type, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXP3UIPROC)(GLenum type, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXP3UIVPROC)(GLenum type, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXP4UIPROC)(GLenum type, GLuint value);
    typedef void (FAN_API_PTR* PFNGLVERTEXP4UIVPROC)(GLenum type, const GLuint* value);
    typedef void (FAN_API_PTR* PFNGLVERTEXPOINTERPROC)(GLint size, GLenum type, GLsizei stride, const void* pointer);
    typedef void (FAN_API_PTR* PFNGLVIEWPORTPROC)(GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (FAN_API_PTR* PFNGLVIEWPORTARRAYVPROC)(GLuint first, GLsizei count, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLVIEWPORTINDEXEDFPROC)(GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h);
    typedef void (FAN_API_PTR* PFNGLVIEWPORTINDEXEDFVPROC)(GLuint index, const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLWAITSYNCPROC)(GLsync sync, GLbitfield flags, GLuint64 timeout);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2DPROC)(GLdouble x, GLdouble y);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2FPROC)(GLfloat x, GLfloat y);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2IPROC)(GLint x, GLint y);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2SPROC)(GLshort x, GLshort y);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS2SVPROC)(const GLshort* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3DPROC)(GLdouble x, GLdouble y, GLdouble z);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3DVPROC)(const GLdouble* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3FPROC)(GLfloat x, GLfloat y, GLfloat z);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3FVPROC)(const GLfloat* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3IPROC)(GLint x, GLint y, GLint z);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3IVPROC)(const GLint* v);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3SPROC)(GLshort x, GLshort y, GLshort z);
    typedef void (FAN_API_PTR* PFNGLWINDOWPOS3SVPROC)(const GLshort* v);

  }
}
    #if defined(fan_platform_windows)
      #include <Windows.h>
        #undef min
        #undef max
    #elif defined(fan_platform_unix)

      #include <X11/X.h>
      #include <X11/Xlib.h>
      #include <X11/Xutil.h>
    #endif

namespace fan {
  namespace opengl {

    #if defined(fan_platform_windows)

    namespace wgl {
      #ifdef __cplusplus
        extern "C" {
        #endif

        /*
        ** Copyright 2013-2020 The Khronos Group Inc.
        ** SPDX-License-Identifier: MIT
        **
        ** This header is generated from the Khronos OpenGL / OpenGL ES XML
        ** API Registry. The current version of the Registry, generator scripts
        ** used to make the header, and the header can be found at
        **   https://github.com/KhronosGroup/OpenGL-Registry
        */

        #if defined(_WIN32) && !defined(APIENTRY) && !defined(__CYGWIN__) && !defined(__SCITECH_SNAP__)
        #define WIN32_LEAN_AND_MEAN 1
        #include <windows.h>
        #endif

        /* Generated on date 20211115 */

        /* Generated C header for:
         * API: wgl
         * Versions considered: .*
         * Versions emitted: .*
         * Default extensions included: wgl
         * Additional extensions included: _nomatch_^
         * Extensions removed: _nomatch_^
         */

        #ifndef WGL_VERSION_1_0
        #define WGL_VERSION_1_0 1
        #define WGL_FONT_LINES                    0
        #define WGL_FONT_POLYGONS                 1
        #define WGL_SWAP_MAIN_PLANE               0x00000001
        #define WGL_SWAP_OVERLAY1                 0x00000002
        #define WGL_SWAP_OVERLAY2                 0x00000004
        #define WGL_SWAP_OVERLAY3                 0x00000008
        #define WGL_SWAP_OVERLAY4                 0x00000010
        #define WGL_SWAP_OVERLAY5                 0x00000020
        #define WGL_SWAP_OVERLAY6                 0x00000040
        #define WGL_SWAP_OVERLAY7                 0x00000080
        #define WGL_SWAP_OVERLAY8                 0x00000100
        #define WGL_SWAP_OVERLAY9                 0x00000200
        #define WGL_SWAP_OVERLAY10                0x00000400
        #define WGL_SWAP_OVERLAY11                0x00000800
        #define WGL_SWAP_OVERLAY12                0x00001000
        #define WGL_SWAP_OVERLAY13                0x00002000
        #define WGL_SWAP_OVERLAY14                0x00004000
        #define WGL_SWAP_OVERLAY15                0x00008000
        #define WGL_SWAP_UNDERLAY1                0x00010000
        #define WGL_SWAP_UNDERLAY2                0x00020000
        #define WGL_SWAP_UNDERLAY3                0x00040000
        #define WGL_SWAP_UNDERLAY4                0x00080000
        #define WGL_SWAP_UNDERLAY5                0x00100000
        #define WGL_SWAP_UNDERLAY6                0x00200000
        #define WGL_SWAP_UNDERLAY7                0x00400000
        #define WGL_SWAP_UNDERLAY8                0x00800000
        #define WGL_SWAP_UNDERLAY9                0x01000000
        #define WGL_SWAP_UNDERLAY10               0x02000000
        #define WGL_SWAP_UNDERLAY11               0x04000000
        #define WGL_SWAP_UNDERLAY12               0x08000000
        #define WGL_SWAP_UNDERLAY13               0x10000000
        #define WGL_SWAP_UNDERLAY14               0x20000000
        #define WGL_SWAP_UNDERLAY15               0x40000000
        typedef int (WINAPI * PFNCHOOSEPIXELFORMATPROC) (HDC hDc, const PIXELFORMATDESCRIPTOR *pPfd);
        typedef int (WINAPI * PFNDESCRIBEPIXELFORMATPROC) (HDC hdc, int ipfd, UINT cjpfd, const PIXELFORMATDESCRIPTOR *ppfd);
        typedef UINT (WINAPI * PFNGETENHMETAFILEPIXELFORMATPROC) (HENHMETAFILE hemf, const PIXELFORMATDESCRIPTOR *ppfd);
        typedef int (WINAPI * PFNGETPIXELFORMATPROC) (HDC hdc);
        typedef BOOL (WINAPI * PFNSETPIXELFORMATPROC) (HDC hdc, int ipfd, const PIXELFORMATDESCRIPTOR *ppfd);
        typedef BOOL (WINAPI * PFNSWAPBUFFERSPROC) (HDC hdc);
        typedef BOOL (WINAPI * PFNWGLCOPYCONTEXTPROC) (HGLRC hglrcSrc, HGLRC hglrcDst, UINT mask);
        typedef HGLRC (WINAPI * PFNWGLCREATECONTEXTPROC) (HDC hDc);
        typedef HGLRC (WINAPI * PFNWGLCREATELAYERCONTEXTPROC) (HDC hDc, int level);
        typedef BOOL (WINAPI * PFNWGLDELETECONTEXTPROC) (HGLRC oldContext);
        typedef BOOL (WINAPI * PFNWGLDESCRIBELAYERPLANEPROC) (HDC hDc, int pixelFormat, int layerPlane, UINT nBytes, const LAYERPLANEDESCRIPTOR *plpd);
        typedef HGLRC (WINAPI * PFNWGLGETCURRENTCONTEXTPROC) (void);
        typedef HDC (WINAPI * PFNWGLGETCURRENTDCPROC) (void);
        typedef int (WINAPI * PFNWGLGETLAYERPALETTEENTRIESPROC) (HDC hdc, int iLayerPlane, int iStart, int cEntries, const COLORREF *pcr);
        typedef PROC (WINAPI * PFNWGLGETPROCADDRESSPROC) (LPCSTR lpszProc);
        typedef BOOL (WINAPI * PFNWGLMAKECURRENTPROC) (HDC hDc, HGLRC newContext);
        typedef BOOL (WINAPI * PFNWGLREALIZELAYERPALETTEPROC) (HDC hdc, int iLayerPlane, BOOL bRealize);
        typedef int (WINAPI * PFNWGLSETLAYERPALETTEENTRIESPROC) (HDC hdc, int iLayerPlane, int iStart, int cEntries, const COLORREF *pcr);
        typedef BOOL (WINAPI * PFNWGLSHARELISTSPROC) (HGLRC hrcSrvShare, HGLRC hrcSrvSource);
        typedef BOOL (WINAPI * PFNWGLSWAPLAYERBUFFERSPROC) (HDC hdc, UINT fuFlags);
        typedef BOOL (WINAPI * PFNWGLUSEFONTBITMAPSPROC) (HDC hDC, DWORD first, DWORD count, DWORD listBase);
        typedef BOOL (WINAPI * PFNWGLUSEFONTBITMAPSAPROC) (HDC hDC, DWORD first, DWORD count, DWORD listBase);
        typedef BOOL (WINAPI * PFNWGLUSEFONTBITMAPSWPROC) (HDC hDC, DWORD first, DWORD count, DWORD listBase);
        typedef BOOL (WINAPI * PFNWGLUSEFONTOUTLINESPROC) (HDC hDC, DWORD first, DWORD count, DWORD listBase, FLOAT deviation, FLOAT extrusion, int format, LPGLYPHMETRICSFLOAT lpgmf);
        typedef BOOL (WINAPI * PFNWGLUSEFONTOUTLINESAPROC) (HDC hDC, DWORD first, DWORD count, DWORD listBase, FLOAT deviation, FLOAT extrusion, int format, LPGLYPHMETRICSFLOAT lpgmf);
        typedef BOOL (WINAPI * PFNWGLUSEFONTOUTLINESWPROC) (HDC hDC, DWORD first, DWORD count, DWORD listBase, FLOAT deviation, FLOAT extrusion, int format, LPGLYPHMETRICSFLOAT lpgmf);
        #ifdef WGL_WGLEXT_PROTOTYPES
        int WINAPI ChoosePixelFormat (HDC hDc, const PIXELFORMATDESCRIPTOR *pPfd);
        int WINAPI DescribePixelFormat (HDC hdc, int ipfd, UINT cjpfd, const PIXELFORMATDESCRIPTOR *ppfd);
        UINT WINAPI GetEnhMetaFilePixelFormat (HENHMETAFILE hemf, const PIXELFORMATDESCRIPTOR *ppfd);
        int WINAPI GetPixelFormat (HDC hdc);
        BOOL WINAPI SetPixelFormat (HDC hdc, int ipfd, const PIXELFORMATDESCRIPTOR *ppfd);
        BOOL WINAPI SwapBuffers (HDC hdc);
        BOOL WINAPI wglCopyContext (HGLRC hglrcSrc, HGLRC hglrcDst, UINT mask);
        HGLRC WINAPI wglCreateContext (HDC hDc);
        HGLRC WINAPI wglCreateLayerContext (HDC hDc, int level);
        BOOL WINAPI wglDeleteContext (HGLRC oldContext);
        BOOL WINAPI wglDescribeLayerPlane (HDC hDc, int pixelFormat, int layerPlane, UINT nBytes, const LAYERPLANEDESCRIPTOR *plpd);
        HGLRC WINAPI wglGetCurrentContext (void);
        HDC WINAPI wglGetCurrentDC (void);
        int WINAPI wglGetLayerPaletteEntries (HDC hdc, int iLayerPlane, int iStart, int cEntries, const COLORREF *pcr);
        PROC WINAPI wglGetProcAddress (LPCSTR lpszProc);
        BOOL WINAPI wglMakeCurrent (HDC hDc, HGLRC newContext);
        BOOL WINAPI wglRealizeLayerPalette (HDC hdc, int iLayerPlane, BOOL bRealize);
        int WINAPI wglSetLayerPaletteEntries (HDC hdc, int iLayerPlane, int iStart, int cEntries, const COLORREF *pcr);
        BOOL WINAPI wglShareLists (HGLRC hrcSrvShare, HGLRC hrcSrvSource);
        BOOL WINAPI wglSwapLayerBuffers (HDC hdc, UINT fuFlags);
        BOOL WINAPI wglUseFontBitmaps (HDC hDC, DWORD first, DWORD count, DWORD listBase);
        BOOL WINAPI wglUseFontBitmapsA (HDC hDC, DWORD first, DWORD count, DWORD listBase);
        BOOL WINAPI wglUseFontBitmapsW (HDC hDC, DWORD first, DWORD count, DWORD listBase);
        BOOL WINAPI wglUseFontOutlines (HDC hDC, DWORD first, DWORD count, DWORD listBase, FLOAT deviation, FLOAT extrusion, int format, LPGLYPHMETRICSFLOAT lpgmf);
        BOOL WINAPI wglUseFontOutlinesA (HDC hDC, DWORD first, DWORD count, DWORD listBase, FLOAT deviation, FLOAT extrusion, int format, LPGLYPHMETRICSFLOAT lpgmf);
        BOOL WINAPI wglUseFontOutlinesW (HDC hDC, DWORD first, DWORD count, DWORD listBase, FLOAT deviation, FLOAT extrusion, int format, LPGLYPHMETRICSFLOAT lpgmf);
        #endif
        #endif /* WGL_VERSION_1_0 */

        #ifndef WGL_ARB_buffer_region
        #define WGL_ARB_buffer_region 1
        #define WGL_FRONT_COLOR_BUFFER_BIT_ARB    0x00000001
        #define WGL_BACK_COLOR_BUFFER_BIT_ARB     0x00000002
        #define WGL_DEPTH_BUFFER_BIT_ARB          0x00000004
        #define WGL_STENCIL_BUFFER_BIT_ARB        0x00000008
    
        typedef HANDLE (WINAPI * PFNWGLCREATEBUFFERREGIONARBPROC) (HDC hDC, int iLayerPlane, UINT uType);
        typedef VOID (WINAPI * PFNWGLDELETEBUFFERREGIONARBPROC) (HANDLE hRegion);
        typedef BOOL (WINAPI * PFNWGLSAVEBUFFERREGIONARBPROC) (HANDLE hRegion, int x, int y, int width, int height);
        typedef BOOL (WINAPI * PFNWGLRESTOREBUFFERREGIONARBPROC) (HANDLE hRegion, int x, int y, int width, int height, int xSrc, int ySrc);
        #ifdef WGL_WGLEXT_PROTOTYPES
        HANDLE WINAPI wglCreateBufferRegionARB (HDC hDC, int iLayerPlane, UINT uType);
        VOID WINAPI wglDeleteBufferRegionARB (HANDLE hRegion);
        BOOL WINAPI wglSaveBufferRegionARB (HANDLE hRegion, int x, int y, int width, int height);
        BOOL WINAPI wglRestoreBufferRegionARB (HANDLE hRegion, int x, int y, int width, int height, int xSrc, int ySrc);
        #endif
        #endif /* WGL_ARB_buffer_region */

        #ifndef WGL_ARB_context_flush_control
        #define WGL_ARB_context_flush_control 1
        #define WGL_CONTEXT_RELEASE_BEHAVIOR_ARB  0x2097
        #define WGL_CONTEXT_RELEASE_BEHAVIOR_NONE_ARB 0
        #define WGL_CONTEXT_RELEASE_BEHAVIOR_FLUSH_ARB 0x2098
        #endif /* WGL_ARB_context_flush_control */

        #ifndef WGL_ARB_create_context
        #define WGL_ARB_create_context 1
        #define WGL_CONTEXT_DEBUG_BIT_ARB         0x00000001
        #define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
        #define WGL_CONTEXT_MAJOR_VERSION_ARB     0x2091
        #define WGL_CONTEXT_MINOR_VERSION_ARB     0x2092
        #define WGL_CONTEXT_LAYER_PLANE_ARB       0x2093
        #define WGL_CONTEXT_FLAGS_ARB             0x2094
        #define ERROR_INVALID_VERSION_ARB         0x2095
        typedef HGLRC (WINAPI * PFNWGLCREATECONTEXTATTRIBSARBPROC) (HDC hDC, HGLRC hShareContext, const int* attribList);
        #ifdef WGL_WGLEXT_PROTOTYPES
        HGLRC WINAPI wglCreateContextAttribsARB (HDC hDC, HGLRC hShareContext, const int *attribList);
        #endif
        #endif /* WGL_ARB_create_context */

        #ifndef WGL_ARB_create_context_no_error
        #define WGL_ARB_create_context_no_error 1
        #define WGL_CONTEXT_OPENGL_NO_ERROR_ARB   0x31B3
        #endif /* WGL_ARB_create_context_no_error */

        #ifndef WGL_ARB_create_context_profile
        #define WGL_ARB_create_context_profile 1
        #define WGL_CONTEXT_PROFILE_MASK_ARB      0x9126
        #define WGL_CONTEXT_CORE_PROFILE_BIT_ARB  0x00000001
        #define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
        #define ERROR_INVALID_PROFILE_ARB         0x2096
        #endif /* WGL_ARB_create_context_profile */

        #ifndef WGL_ARB_create_context_robustness
        #define WGL_ARB_create_context_robustness 1
        #define WGL_CONTEXT_ROBUST_ACCESS_BIT_ARB 0x00000004
        #define WGL_LOSE_CONTEXT_ON_RESET_ARB     0x8252
        #define WGL_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB 0x8256
        #define WGL_NO_RESET_NOTIFICATION_ARB     0x8261
        #endif /* WGL_ARB_create_context_robustness */

        #ifndef WGL_ARB_extensions_string
        #define WGL_ARB_extensions_string 1
        typedef const char *(WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);
        #ifdef WGL_WGLEXT_PROTOTYPES
        const char *WINAPI wglGetExtensionsStringARB (HDC hdc);
        #endif
        #endif /* WGL_ARB_extensions_string */

        #ifndef WGL_ARB_framebuffer_sRGB
        #define WGL_ARB_framebuffer_sRGB 1
        #define WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB  0x20A9
        #endif /* WGL_ARB_framebuffer_sRGB */

        #ifndef WGL_ARB_make_current_read
        #define WGL_ARB_make_current_read 1
        #define ERROR_INVALID_PIXEL_TYPE_ARB      0x2043
        #define ERROR_INCOMPATIBLE_DEVICE_CONTEXTS_ARB 0x2054
        typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTARBPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);
        typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCARBPROC) (void);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglMakeContextCurrentARB (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);
        HDC WINAPI wglGetCurrentReadDCARB (void);
        #endif
        #endif /* WGL_ARB_make_current_read */

        #ifndef WGL_ARB_multisample
        #define WGL_ARB_multisample 1
        #define WGL_SAMPLE_BUFFERS_ARB            0x2041
        #define WGL_SAMPLES_ARB                   0x2042
        #endif /* WGL_ARB_multisample */

        #ifndef WGL_ARB_pbuffer
        #define WGL_ARB_pbuffer 1
        DECLARE_HANDLE(HPBUFFERARB);
        #define WGL_DRAW_TO_PBUFFER_ARB           0x202D
        #define WGL_MAX_PBUFFER_PIXELS_ARB        0x202E
        #define WGL_MAX_PBUFFER_WIDTH_ARB         0x202F
        #define WGL_MAX_PBUFFER_HEIGHT_ARB        0x2030
        #define WGL_PBUFFER_LARGEST_ARB           0x2033
        #define WGL_PBUFFER_WIDTH_ARB             0x2034
        #define WGL_PBUFFER_HEIGHT_ARB            0x2035
        #define WGL_PBUFFER_LOST_ARB              0x2036
        typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int *piAttribList);
        typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
        typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer, HDC hDC);
        typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
        typedef BOOL (WINAPI * PFNWGLQUERYPBUFFERARBPROC) (HPBUFFERARB hPbuffer, int iAttribute, int *piValue);
        #ifdef WGL_WGLEXT_PROTOTYPES
        HPBUFFERARB WINAPI wglCreatePbufferARB (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int *piAttribList);
        HDC WINAPI wglGetPbufferDCARB (HPBUFFERARB hPbuffer);
        int WINAPI wglReleasePbufferDCARB (HPBUFFERARB hPbuffer, HDC hDC);
        BOOL WINAPI wglDestroyPbufferARB (HPBUFFERARB hPbuffer);
        BOOL WINAPI wglQueryPbufferARB (HPBUFFERARB hPbuffer, int iAttribute, int *piValue);
        #endif
        #endif /* WGL_ARB_pbuffer */

        #ifndef WGL_ARB_pixel_format
        #define WGL_ARB_pixel_format 1
        #define WGL_NUMBER_PIXEL_FORMATS_ARB      0x2000
        #define WGL_DRAW_TO_WINDOW_ARB            0x2001
        #define WGL_DRAW_TO_BITMAP_ARB            0x2002
        #define WGL_ACCELERATION_ARB              0x2003
        #define WGL_NEED_PALETTE_ARB              0x2004
        #define WGL_NEED_SYSTEM_PALETTE_ARB       0x2005
        #define WGL_SWAP_LAYER_BUFFERS_ARB        0x2006
        #define WGL_SWAP_METHOD_ARB               0x2007
        #define WGL_NUMBER_OVERLAYS_ARB           0x2008
        #define WGL_NUMBER_UNDERLAYS_ARB          0x2009
        #define WGL_TRANSPARENT_ARB               0x200A
        #define WGL_TRANSPARENT_RED_VALUE_ARB     0x2037
        #define WGL_TRANSPARENT_GREEN_VALUE_ARB   0x2038
        #define WGL_TRANSPARENT_BLUE_VALUE_ARB    0x2039
        #define WGL_TRANSPARENT_ALPHA_VALUE_ARB   0x203A
        #define WGL_TRANSPARENT_INDEX_VALUE_ARB   0x203B
        #define WGL_SHARE_DEPTH_ARB               0x200C
        #define WGL_SHARE_STENCIL_ARB             0x200D
        #define WGL_SHARE_ACCUM_ARB               0x200E
        #define WGL_SUPPORT_GDI_ARB               0x200F
        #define WGL_SUPPORT_OPENGL_ARB            0x2010
        #define WGL_DOUBLE_BUFFER_ARB             0x2011
        #define WGL_STEREO_ARB                    0x2012
        #define WGL_PIXEL_TYPE_ARB                0x2013
        #define WGL_COLOR_BITS_ARB                0x2014
        #define WGL_RED_BITS_ARB                  0x2015
        #define WGL_RED_SHIFT_ARB                 0x2016
        #define WGL_GREEN_BITS_ARB                0x2017
        #define WGL_GREEN_SHIFT_ARB               0x2018
        #define WGL_BLUE_BITS_ARB                 0x2019
        #define WGL_BLUE_SHIFT_ARB                0x201A
        #define WGL_ALPHA_BITS_ARB                0x201B
        #define WGL_ALPHA_SHIFT_ARB               0x201C
        #define WGL_ACCUM_BITS_ARB                0x201D
        #define WGL_ACCUM_RED_BITS_ARB            0x201E
        #define WGL_ACCUM_GREEN_BITS_ARB          0x201F
        #define WGL_ACCUM_BLUE_BITS_ARB           0x2020
        #define WGL_ACCUM_ALPHA_BITS_ARB          0x2021
        #define WGL_DEPTH_BITS_ARB                0x2022
        #define WGL_STENCIL_BITS_ARB              0x2023
        #define WGL_AUX_BUFFERS_ARB               0x2024
        #define WGL_NO_ACCELERATION_ARB           0x2025
        #define WGL_GENERIC_ACCELERATION_ARB      0x2026
        #define WGL_FULL_ACCELERATION_ARB         0x2027
        #define WGL_SWAP_EXCHANGE_ARB             0x2028
        #define WGL_SWAP_COPY_ARB                 0x2029
        #define WGL_SWAP_UNDEFINED_ARB            0x202A
        #define WGL_TYPE_RGBA_ARB                 0x202B
        #define WGL_TYPE_COLORINDEX_ARB           0x202C
        typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (HDC hdc, const int* piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
        typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, FLOAT *pfValues);
        typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetPixelFormatAttribivARB (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, int *piValues);
        BOOL WINAPI wglGetPixelFormatAttribfvARB (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int *piAttributes, FLOAT *pfValues);
        BOOL WINAPI wglChoosePixelFormatARB (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
        #endif
        #endif /* WGL_ARB_pixel_format */

        #ifndef WGL_ARB_pixel_format_float
        #define WGL_ARB_pixel_format_float 1
        #define WGL_TYPE_RGBA_FLOAT_ARB           0x21A0
        #endif /* WGL_ARB_pixel_format_float */

        #ifndef WGL_ARB_render_texture
        #define WGL_ARB_render_texture 1
        #define WGL_BIND_TO_TEXTURE_RGB_ARB       0x2070
        #define WGL_BIND_TO_TEXTURE_RGBA_ARB      0x2071
        #define WGL_TEXTURE_FORMAT_ARB            0x2072
        #define WGL_TEXTURE_TARGET_ARB            0x2073
        #define WGL_MIPMAP_TEXTURE_ARB            0x2074
        #define WGL_TEXTURE_RGB_ARB               0x2075
        #define WGL_TEXTURE_RGBA_ARB              0x2076
        #define WGL_NO_TEXTURE_ARB                0x2077
        #define WGL_TEXTURE_CUBE_MAP_ARB          0x2078
        #define WGL_TEXTURE_1D_ARB                0x2079
        #define WGL_TEXTURE_2D_ARB                0x207A
        #define WGL_MIPMAP_LEVEL_ARB              0x207B
        #define WGL_CUBE_MAP_FACE_ARB             0x207C
        #define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
        #define WGL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x207E
        #define WGL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x207F
        #define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x2080
        #define WGL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x2081
        #define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x2082
        #define WGL_FRONT_LEFT_ARB                0x2083
        #define WGL_FRONT_RIGHT_ARB               0x2084
        #define WGL_BACK_LEFT_ARB                 0x2085
        #define WGL_BACK_RIGHT_ARB                0x2086
        #define WGL_AUX0_ARB                      0x2087
        #define WGL_AUX1_ARB                      0x2088
        #define WGL_AUX2_ARB                      0x2089
        #define WGL_AUX3_ARB                      0x208A
        #define WGL_AUX4_ARB                      0x208B
        #define WGL_AUX5_ARB                      0x208C
        #define WGL_AUX6_ARB                      0x208D
        #define WGL_AUX7_ARB                      0x208E
        #define WGL_AUX8_ARB                      0x208F
        #define WGL_AUX9_ARB                      0x2090
        typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
        typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
        typedef BOOL (WINAPI * PFNWGLSETPBUFFERATTRIBARBPROC) (HPBUFFERARB hPbuffer, const int *piAttribList);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglBindTexImageARB (HPBUFFERARB hPbuffer, int iBuffer);
        BOOL WINAPI wglReleaseTexImageARB (HPBUFFERARB hPbuffer, int iBuffer);
        BOOL WINAPI wglSetPbufferAttribARB (HPBUFFERARB hPbuffer, const int *piAttribList);
        #endif
        #endif /* WGL_ARB_render_texture */

        #ifndef WGL_ARB_robustness_application_isolation
        #define WGL_ARB_robustness_application_isolation 1
        #define WGL_CONTEXT_RESET_ISOLATION_BIT_ARB 0x00000008
        #endif /* WGL_ARB_robustness_application_isolation */

        #ifndef WGL_ARB_robustness_share_group_isolation
        #define WGL_ARB_robustness_share_group_isolation 1
        #endif /* WGL_ARB_robustness_share_group_isolation */

        #ifndef WGL_3DFX_multisample
        #define WGL_3DFX_multisample 1
        #define WGL_SAMPLE_BUFFERS_3DFX           0x2060
        #define WGL_SAMPLES_3DFX                  0x2061
        #endif /* WGL_3DFX_multisample */

        #ifndef WGL_3DL_stereo_control
        #define WGL_3DL_stereo_control 1
        #define WGL_STEREO_EMITTER_ENABLE_3DL     0x2055
        #define WGL_STEREO_EMITTER_DISABLE_3DL    0x2056
        #define WGL_STEREO_POLARITY_NORMAL_3DL    0x2057
        #define WGL_STEREO_POLARITY_INVERT_3DL    0x2058
        typedef BOOL (WINAPI * PFNWGLSETSTEREOEMITTERSTATE3DLPROC) (HDC hDC, UINT uState);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglSetStereoEmitterState3DL (HDC hDC, UINT uState);
        #endif
        #endif /* WGL_3DL_stereo_control */

        #ifndef WGL_AMD_gpu_association
        #define WGL_AMD_gpu_association 1
        #define WGL_GPU_VENDOR_AMD                0x1F00
        #define WGL_GPU_RENDERER_STRING_AMD       0x1F01
        #define WGL_GPU_OPENGL_VERSION_STRING_AMD 0x1F02
        #define WGL_GPU_FASTEST_TARGET_GPUS_AMD   0x21A2
        #define WGL_GPU_RAM_AMD                   0x21A3
        #define WGL_GPU_CLOCK_AMD                 0x21A4
        #define WGL_GPU_NUM_PIPES_AMD             0x21A5
        #define WGL_GPU_NUM_SIMD_AMD              0x21A6
        #define WGL_GPU_NUM_RB_AMD                0x21A7
        #define WGL_GPU_NUM_SPI_AMD               0x21A8
        typedef UINT (WINAPI * PFNWGLGETGPUIDSAMDPROC) (UINT maxCount, UINT *ids);
        typedef INT (WINAPI * PFNWGLGETGPUINFOAMDPROC) (UINT id, INT property, GLenum dataType, UINT size, void *data);
        typedef UINT (WINAPI * PFNWGLGETCONTEXTGPUIDAMDPROC) (HGLRC hglrc);
        typedef HGLRC (WINAPI * PFNWGLCREATEASSOCIATEDCONTEXTAMDPROC) (UINT id);
        typedef HGLRC (WINAPI * PFNWGLCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC) (UINT id, HGLRC hShareContext, const int *attribList);
        typedef BOOL (WINAPI * PFNWGLDELETEASSOCIATEDCONTEXTAMDPROC) (HGLRC hglrc);
        typedef BOOL (WINAPI * PFNWGLMAKEASSOCIATEDCONTEXTCURRENTAMDPROC) (HGLRC hglrc);
        typedef HGLRC (WINAPI * PFNWGLGETCURRENTASSOCIATEDCONTEXTAMDPROC) (void);
        typedef VOID (WINAPI * PFNWGLBLITCONTEXTFRAMEBUFFERAMDPROC) (HGLRC dstCtx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum image_filter);
        #ifdef WGL_WGLEXT_PROTOTYPES
        UINT WINAPI wglGetGPUIDsAMD (UINT maxCount, UINT *ids);
        INT WINAPI wglGetGPUInfoAMD (UINT id, INT property, GLenum dataType, UINT size, void *data);
        UINT WINAPI wglGetContextGPUIDAMD (HGLRC hglrc);
        HGLRC WINAPI wglCreateAssociatedContextAMD (UINT id);
        HGLRC WINAPI wglCreateAssociatedContextAttribsAMD (UINT id, HGLRC hShareContext, const int *attribList);
        BOOL WINAPI wglDeleteAssociatedContextAMD (HGLRC hglrc);
        BOOL WINAPI wglMakeAssociatedContextCurrentAMD (HGLRC hglrc);
        HGLRC WINAPI wglGetCurrentAssociatedContextAMD (void);
        VOID WINAPI wglBlitContextFramebufferAMD (HGLRC dstCtx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
        #endif
        #endif /* WGL_AMD_gpu_association */

        #ifndef WGL_ATI_pixel_format_float
        #define WGL_ATI_pixel_format_float 1
        #define WGL_TYPE_RGBA_FLOAT_ATI           0x21A0
        #endif /* WGL_ATI_pixel_format_float */

        #ifndef WGL_ATI_render_texture_rectangle
        #define WGL_ATI_render_texture_rectangle 1
        #define WGL_TEXTURE_RECTANGLE_ATI         0x21A5
        #endif /* WGL_ATI_render_texture_rectangle */

        #ifndef WGL_EXT_colorspace
        #define WGL_EXT_colorspace 1
        #define WGL_COLORSPACE_EXT                0x309D
        #define WGL_COLORSPACE_SRGB_EXT           0x3089
        #define WGL_COLORSPACE_LINEAR_EXT         0x308A
        #endif /* WGL_EXT_colorspace */

        #ifndef WGL_EXT_create_context_es2_profile
        #define WGL_EXT_create_context_es2_profile 1
        #define WGL_CONTEXT_ES2_PROFILE_BIT_EXT   0x00000004
        #endif /* WGL_EXT_create_context_es2_profile */

        #ifndef WGL_EXT_create_context_es_profile
        #define WGL_EXT_create_context_es_profile 1
        #define WGL_CONTEXT_ES_PROFILE_BIT_EXT    0x00000004
        #endif /* WGL_EXT_create_context_es_profile */

        #ifndef WGL_EXT_depth_float
        #define WGL_EXT_depth_float 1
        #define WGL_DEPTH_FLOAT_EXT               0x2040
        #endif /* WGL_EXT_depth_float */

        #ifndef WGL_EXT_display_color_table
        #define WGL_EXT_display_color_table 1
        typedef GLboolean (WINAPI * PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC) (GLushort id);
        typedef GLboolean (WINAPI * PFNWGLLOADDISPLAYCOLORTABLEEXTPROC) (const GLushort *table, GLuint length);
        typedef GLboolean (WINAPI * PFNWGLBINDDISPLAYCOLORTABLEEXTPROC) (GLushort id);
        typedef VOID (WINAPI * PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC) (GLushort id);
        #ifdef WGL_WGLEXT_PROTOTYPES
        GLboolean WINAPI wglCreateDisplayColorTableEXT (GLushort id);
        GLboolean WINAPI wglLoadDisplayColorTableEXT (const GLushort *table, GLuint length);
        GLboolean WINAPI wglBindDisplayColorTableEXT (GLushort id);
        VOID WINAPI wglDestroyDisplayColorTableEXT (GLushort id);
        #endif
        #endif /* WGL_EXT_display_color_table */

        #ifndef WGL_EXT_extensions_string
        #define WGL_EXT_extensions_string 1
        typedef const char *(WINAPI * PFNWGLGETEXTENSIONSSTRINGEXTPROC) (void);
        #ifdef WGL_WGLEXT_PROTOTYPES
        const char *WINAPI wglGetExtensionsStringEXT (void);
        #endif
        #endif /* WGL_EXT_extensions_string */

        #ifndef WGL_EXT_framebuffer_sRGB
        #define WGL_EXT_framebuffer_sRGB 1
        #define WGL_FRAMEBUFFER_SRGB_CAPABLE_EXT  0x20A9
        #endif /* WGL_EXT_framebuffer_sRGB */

        #ifndef WGL_EXT_make_current_read
        #define WGL_EXT_make_current_read 1
        #define ERROR_INVALID_PIXEL_TYPE_EXT      0x2043
        typedef BOOL (WINAPI * PFNWGLMAKECONTEXTCURRENTEXTPROC) (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);
        typedef HDC (WINAPI * PFNWGLGETCURRENTREADDCEXTPROC) (void);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglMakeContextCurrentEXT (HDC hDrawDC, HDC hReadDC, HGLRC hglrc);
        HDC WINAPI wglGetCurrentReadDCEXT (void);
        #endif
        #endif /* WGL_EXT_make_current_read */

        #ifndef WGL_EXT_multisample
        #define WGL_EXT_multisample 1
        #define WGL_SAMPLE_BUFFERS_EXT            0x2041
        #define WGL_SAMPLES_EXT                   0x2042
        #endif /* WGL_EXT_multisample */

        #ifndef WGL_EXT_pbuffer
        #define WGL_EXT_pbuffer 1
        DECLARE_HANDLE(HPBUFFEREXT);
        #define WGL_DRAW_TO_PBUFFER_EXT           0x202D
        #define WGL_MAX_PBUFFER_PIXELS_EXT        0x202E
        #define WGL_MAX_PBUFFER_WIDTH_EXT         0x202F
        #define WGL_MAX_PBUFFER_HEIGHT_EXT        0x2030
        #define WGL_OPTIMAL_PBUFFER_WIDTH_EXT     0x2031
        #define WGL_OPTIMAL_PBUFFER_HEIGHT_EXT    0x2032
        #define WGL_PBUFFER_LARGEST_EXT           0x2033
        #define WGL_PBUFFER_WIDTH_EXT             0x2034
        #define WGL_PBUFFER_HEIGHT_EXT            0x2035
        typedef HPBUFFEREXT (WINAPI * PFNWGLCREATEPBUFFEREXTPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int *piAttribList);
        typedef HDC (WINAPI * PFNWGLGETPBUFFERDCEXTPROC) (HPBUFFEREXT hPbuffer);
        typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCEXTPROC) (HPBUFFEREXT hPbuffer, HDC hDC);
        typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFEREXTPROC) (HPBUFFEREXT hPbuffer);
        typedef BOOL (WINAPI * PFNWGLQUERYPBUFFEREXTPROC) (HPBUFFEREXT hPbuffer, int iAttribute, int *piValue);
        #ifdef WGL_WGLEXT_PROTOTYPES
        HPBUFFEREXT WINAPI wglCreatePbufferEXT (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int *piAttribList);
        HDC WINAPI wglGetPbufferDCEXT (HPBUFFEREXT hPbuffer);
        int WINAPI wglReleasePbufferDCEXT (HPBUFFEREXT hPbuffer, HDC hDC);
        BOOL WINAPI wglDestroyPbufferEXT (HPBUFFEREXT hPbuffer);
        BOOL WINAPI wglQueryPbufferEXT (HPBUFFEREXT hPbuffer, int iAttribute, int *piValue);
        #endif
        #endif /* WGL_EXT_pbuffer */

        #ifndef WGL_EXT_pixel_format
        #define WGL_EXT_pixel_format 1
        #define WGL_NUMBER_PIXEL_FORMATS_EXT      0x2000
        #define WGL_DRAW_TO_WINDOW_EXT            0x2001
        #define WGL_DRAW_TO_BITMAP_EXT            0x2002
        #define WGL_ACCELERATION_EXT              0x2003
        #define WGL_NEED_PALETTE_EXT              0x2004
        #define WGL_NEED_SYSTEM_PALETTE_EXT       0x2005
        #define WGL_SWAP_LAYER_BUFFERS_EXT        0x2006
        #define WGL_SWAP_METHOD_EXT               0x2007
        #define WGL_NUMBER_OVERLAYS_EXT           0x2008
        #define WGL_NUMBER_UNDERLAYS_EXT          0x2009
        #define WGL_TRANSPARENT_EXT               0x200A
        #define WGL_TRANSPARENT_VALUE_EXT         0x200B
        #define WGL_SHARE_DEPTH_EXT               0x200C
        #define WGL_SHARE_STENCIL_EXT             0x200D
        #define WGL_SHARE_ACCUM_EXT               0x200E
        #define WGL_SUPPORT_GDI_EXT               0x200F
        #define WGL_SUPPORT_OPENGL_EXT            0x2010
        #define WGL_DOUBLE_BUFFER_EXT             0x2011
        #define WGL_STEREO_EXT                    0x2012
        #define WGL_PIXEL_TYPE_EXT                0x2013
        #define WGL_COLOR_BITS_EXT                0x2014
        #define WGL_RED_BITS_EXT                  0x2015
        #define WGL_RED_SHIFT_EXT                 0x2016
        #define WGL_GREEN_BITS_EXT                0x2017
        #define WGL_GREEN_SHIFT_EXT               0x2018
        #define WGL_BLUE_BITS_EXT                 0x2019
        #define WGL_BLUE_SHIFT_EXT                0x201A
        #define WGL_ALPHA_BITS_EXT                0x201B
        #define WGL_ALPHA_SHIFT_EXT               0x201C
        #define WGL_ACCUM_BITS_EXT                0x201D
        #define WGL_ACCUM_RED_BITS_EXT            0x201E
        #define WGL_ACCUM_GREEN_BITS_EXT          0x201F
        #define WGL_ACCUM_BLUE_BITS_EXT           0x2020
        #define WGL_ACCUM_ALPHA_BITS_EXT          0x2021
        #define WGL_DEPTH_BITS_EXT                0x2022
        #define WGL_STENCIL_BITS_EXT              0x2023
        #define WGL_AUX_BUFFERS_EXT               0x2024
        #define WGL_NO_ACCELERATION_EXT           0x2025
        #define WGL_GENERIC_ACCELERATION_EXT      0x2026
        #define WGL_FULL_ACCELERATION_EXT         0x2027
        #define WGL_SWAP_EXCHANGE_EXT             0x2028
        #define WGL_SWAP_COPY_EXT                 0x2029
        #define WGL_SWAP_UNDEFINED_EXT            0x202A
        #define WGL_TYPE_RGBA_EXT                 0x202B
        #define WGL_TYPE_COLORINDEX_EXT           0x202C
        typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVEXTPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int *piAttributes, int *piValues);
        typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVEXTPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int *piAttributes, FLOAT *pfValues);
        typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATEXTPROC) (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetPixelFormatAttribivEXT (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int *piAttributes, int *piValues);
        BOOL WINAPI wglGetPixelFormatAttribfvEXT (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, int *piAttributes, FLOAT *pfValues);
        BOOL WINAPI wglChoosePixelFormatEXT (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
        #endif
        #endif /* WGL_EXT_pixel_format */

        #ifndef WGL_EXT_pixel_format_packed_float
        #define WGL_EXT_pixel_format_packed_float 1
        #define WGL_TYPE_RGBA_UNSIGNED_FLOAT_EXT  0x20A8
        #endif /* WGL_EXT_pixel_format_packed_float */

        #ifndef WGL_EXT_swap_control
        #define WGL_EXT_swap_control 1
        typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
        typedef int (WINAPI * PFNWGLGETSWAPINTERVALEXTPROC) (void);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglSwapIntervalEXT (int interval);
        int WINAPI wglGetSwapIntervalEXT (void);
        #endif
        #endif /* WGL_EXT_swap_control */

        #ifndef WGL_EXT_swap_control_tear
        #define WGL_EXT_swap_control_tear 1
        #endif /* WGL_EXT_swap_control_tear */

        #ifndef WGL_I3D_digital_video_control
        #define WGL_I3D_digital_video_control 1
        #define WGL_DIGITAL_VIDEO_CURSOR_ALPHA_FRAMEBUFFER_I3D 0x2050
        #define WGL_DIGITAL_VIDEO_CURSOR_ALPHA_VALUE_I3D 0x2051
        #define WGL_DIGITAL_VIDEO_CURSOR_INCLUDED_I3D 0x2052
        #define WGL_DIGITAL_VIDEO_GAMMA_CORRECTED_I3D 0x2053
        typedef BOOL (WINAPI * PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC) (HDC hDC, int iAttribute, int *piValue);
        typedef BOOL (WINAPI * PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC) (HDC hDC, int iAttribute, const int *piValue);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetDigitalVideoParametersI3D (HDC hDC, int iAttribute, int *piValue);
        BOOL WINAPI wglSetDigitalVideoParametersI3D (HDC hDC, int iAttribute, const int *piValue);
        #endif
        #endif /* WGL_I3D_digital_video_control */

        #ifndef WGL_I3D_gamma
        #define WGL_I3D_gamma 1
        #define WGL_GAMMA_TABLE_SIZE_I3D          0x204E
        #define WGL_GAMMA_EXCLUDE_DESKTOP_I3D     0x204F
        typedef BOOL (WINAPI * PFNWGLGETGAMMATABLEPARAMETERSI3DPROC) (HDC hDC, int iAttribute, int *piValue);
        typedef BOOL (WINAPI * PFNWGLSETGAMMATABLEPARAMETERSI3DPROC) (HDC hDC, int iAttribute, const int *piValue);
        typedef BOOL (WINAPI * PFNWGLGETGAMMATABLEI3DPROC) (HDC hDC, int iEntries, USHORT *puRed, USHORT *puGreen, USHORT *puBlue);
        typedef BOOL (WINAPI * PFNWGLSETGAMMATABLEI3DPROC) (HDC hDC, int iEntries, const USHORT *puRed, const USHORT *puGreen, const USHORT *puBlue);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetGammaTableParametersI3D (HDC hDC, int iAttribute, int *piValue);
        BOOL WINAPI wglSetGammaTableParametersI3D (HDC hDC, int iAttribute, const int *piValue);
        BOOL WINAPI wglGetGammaTableI3D (HDC hDC, int iEntries, USHORT *puRed, USHORT *puGreen, USHORT *puBlue);
        BOOL WINAPI wglSetGammaTableI3D (HDC hDC, int iEntries, const USHORT *puRed, const USHORT *puGreen, const USHORT *puBlue);
        #endif
        #endif /* WGL_I3D_gamma */

        #ifndef WGL_I3D_genlock
        #define WGL_I3D_genlock 1
        #define WGL_GENLOCK_SOURCE_MULTIVIEW_I3D  0x2044
        #define WGL_GENLOCK_SOURCE_EXTERNAL_SYNC_I3D 0x2045
        #define WGL_GENLOCK_SOURCE_EXTERNAL_FIELD_I3D 0x2046
        #define WGL_GENLOCK_SOURCE_EXTERNAL_TTL_I3D 0x2047
        #define WGL_GENLOCK_SOURCE_DIGITAL_SYNC_I3D 0x2048
        #define WGL_GENLOCK_SOURCE_DIGITAL_FIELD_I3D 0x2049
        #define WGL_GENLOCK_SOURCE_EDGE_FALLING_I3D 0x204A
        #define WGL_GENLOCK_SOURCE_EDGE_RISING_I3D 0x204B
        #define WGL_GENLOCK_SOURCE_EDGE_BOTH_I3D  0x204C
        typedef BOOL (WINAPI * PFNWGLENABLEGENLOCKI3DPROC) (HDC hDC);
        typedef BOOL (WINAPI * PFNWGLDISABLEGENLOCKI3DPROC) (HDC hDC);
        typedef BOOL (WINAPI * PFNWGLISENABLEDGENLOCKI3DPROC) (HDC hDC, BOOL *pFlag);
        typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEI3DPROC) (HDC hDC, UINT uSource);
        typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEI3DPROC) (HDC hDC, UINT *uSource);
        typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEEDGEI3DPROC) (HDC hDC, UINT uEdge);
        typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEEDGEI3DPROC) (HDC hDC, UINT *uEdge);
        typedef BOOL (WINAPI * PFNWGLGENLOCKSAMPLERATEI3DPROC) (HDC hDC, UINT uRate);
        typedef BOOL (WINAPI * PFNWGLGETGENLOCKSAMPLERATEI3DPROC) (HDC hDC, UINT *uRate);
        typedef BOOL (WINAPI * PFNWGLGENLOCKSOURCEDELAYI3DPROC) (HDC hDC, UINT uDelay);
        typedef BOOL (WINAPI * PFNWGLGETGENLOCKSOURCEDELAYI3DPROC) (HDC hDC, UINT *uDelay);
        typedef BOOL (WINAPI * PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC) (HDC hDC, UINT *uMaxLineDelay, UINT *uMaxPixelDelay);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglEnableGenlockI3D (HDC hDC);
        BOOL WINAPI wglDisableGenlockI3D (HDC hDC);
        BOOL WINAPI wglIsEnabledGenlockI3D (HDC hDC, BOOL *pFlag);
        BOOL WINAPI wglGenlockSourceI3D (HDC hDC, UINT uSource);
        BOOL WINAPI wglGetGenlockSourceI3D (HDC hDC, UINT *uSource);
        BOOL WINAPI wglGenlockSourceEdgeI3D (HDC hDC, UINT uEdge);
        BOOL WINAPI wglGetGenlockSourceEdgeI3D (HDC hDC, UINT *uEdge);
        BOOL WINAPI wglGenlockSampleRateI3D (HDC hDC, UINT uRate);
        BOOL WINAPI wglGetGenlockSampleRateI3D (HDC hDC, UINT *uRate);
        BOOL WINAPI wglGenlockSourceDelayI3D (HDC hDC, UINT uDelay);
        BOOL WINAPI wglGetGenlockSourceDelayI3D (HDC hDC, UINT *uDelay);
        BOOL WINAPI wglQueryGenlockMaxSourceDelayI3D (HDC hDC, UINT *uMaxLineDelay, UINT *uMaxPixelDelay);
        #endif
        #endif /* WGL_I3D_genlock */

        #ifndef WGL_I3D_image_buffer
        #define WGL_I3D_image_buffer 1
        #define WGL_IMAGE_BUFFER_MIN_ACCESS_I3D   0x00000001
        #define WGL_IMAGE_BUFFER_LOCK_I3D         0x00000002
        typedef LPVOID (WINAPI * PFNWGLCREATEIMAGEBUFFERI3DPROC) (HDC hDC, DWORD dwSize, UINT uFlags);
        typedef BOOL (WINAPI * PFNWGLDESTROYIMAGEBUFFERI3DPROC) (HDC hDC, LPVOID pAddress);
        typedef BOOL (WINAPI * PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC) (HDC hDC, const HANDLE *pEvent, const LPVOID *pAddress, const DWORD *pSize, UINT count);
        typedef BOOL (WINAPI * PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC) (HDC hDC, const LPVOID *pAddress, UINT count);
        #ifdef WGL_WGLEXT_PROTOTYPES
        LPVOID WINAPI wglCreateImageBufferI3D (HDC hDC, DWORD dwSize, UINT uFlags);
        BOOL WINAPI wglDestroyImageBufferI3D (HDC hDC, LPVOID pAddress);
        BOOL WINAPI wglAssociateImageBufferEventsI3D (HDC hDC, const HANDLE *pEvent, const LPVOID *pAddress, const DWORD *pSize, UINT count);
        BOOL WINAPI wglReleaseImageBufferEventsI3D (HDC hDC, const LPVOID *pAddress, UINT count);
        #endif
        #endif /* WGL_I3D_image_buffer */

        #ifndef WGL_I3D_swap_frame_lock
        #define WGL_I3D_swap_frame_lock 1
        typedef BOOL (WINAPI * PFNWGLENABLEFRAMELOCKI3DPROC) (void);
        typedef BOOL (WINAPI * PFNWGLDISABLEFRAMELOCKI3DPROC) (void);
        typedef BOOL (WINAPI * PFNWGLISENABLEDFRAMELOCKI3DPROC) (BOOL *pFlag);
        typedef BOOL (WINAPI * PFNWGLQUERYFRAMELOCKMASTERI3DPROC) (BOOL *pFlag);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglEnableFrameLockI3D (void);
        BOOL WINAPI wglDisableFrameLockI3D (void);
        BOOL WINAPI wglIsEnabledFrameLockI3D (BOOL *pFlag);
        BOOL WINAPI wglQueryFrameLockMasterI3D (BOOL *pFlag);
        #endif
        #endif /* WGL_I3D_swap_frame_lock */

        #ifndef WGL_I3D_swap_frame_usage
        #define WGL_I3D_swap_frame_usage 1
        typedef BOOL (WINAPI * PFNWGLGETFRAMEUSAGEI3DPROC) (float *pUsage);
        typedef BOOL (WINAPI * PFNWGLBEGINFRAMETRACKINGI3DPROC) (void);
        typedef BOOL (WINAPI * PFNWGLENDFRAMETRACKINGI3DPROC) (void);
        typedef BOOL (WINAPI * PFNWGLQUERYFRAMETRACKINGI3DPROC) (DWORD *pFrameCount, DWORD *pMissedFrames, float *pLastMissedUsage);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetFrameUsageI3D (float *pUsage);
        BOOL WINAPI wglBeginFrameTrackingI3D (void);
        BOOL WINAPI wglEndFrameTrackingI3D (void);
        BOOL WINAPI wglQueryFrameTrackingI3D (DWORD *pFrameCount, DWORD *pMissedFrames, float *pLastMissedUsage);
        #endif
        #endif /* WGL_I3D_swap_frame_usage */

        #ifndef WGL_NV_DX_interop
        #define WGL_NV_DX_interop 1
        #define WGL_ACCESS_READ_ONLY_NV           0x00000000
        #define WGL_ACCESS_READ_WRITE_NV          0x00000001
        #define WGL_ACCESS_WRITE_DISCARD_NV       0x00000002
        typedef BOOL (WINAPI * PFNWGLDXSETRESOURCESHAREHANDLENVPROC) (void *dxObject, HANDLE shareHandle);
        typedef HANDLE (WINAPI * PFNWGLDXOPENDEVICENVPROC) (void *dxDevice);
        typedef BOOL (WINAPI * PFNWGLDXCLOSEDEVICENVPROC) (HANDLE hDevice);
        typedef HANDLE (WINAPI * PFNWGLDXREGISTEROBJECTNVPROC) (HANDLE hDevice, void *dxObject, GLuint name, GLenum type, GLenum access);
        typedef BOOL (WINAPI * PFNWGLDXUNREGISTEROBJECTNVPROC) (HANDLE hDevice, HANDLE hObject);
        typedef BOOL (WINAPI * PFNWGLDXOBJECTACCESSNVPROC) (HANDLE hObject, GLenum access);
        typedef BOOL (WINAPI * PFNWGLDXLOCKOBJECTSNVPROC) (HANDLE hDevice, GLint count, HANDLE *hObjects);
        typedef BOOL (WINAPI * PFNWGLDXUNLOCKOBJECTSNVPROC) (HANDLE hDevice, GLint count, HANDLE *hObjects);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglDXSetResourceShareHandleNV (void *dxObject, HANDLE shareHandle);
        HANDLE WINAPI wglDXOpenDeviceNV (void *dxDevice);
        BOOL WINAPI wglDXCloseDeviceNV (HANDLE hDevice);
        HANDLE WINAPI wglDXRegisterObjectNV (HANDLE hDevice, void *dxObject, GLuint name, GLenum type, GLenum access);
        BOOL WINAPI wglDXUnregisterObjectNV (HANDLE hDevice, HANDLE hObject);
        BOOL WINAPI wglDXObjectAccessNV (HANDLE hObject, GLenum access);
        BOOL WINAPI wglDXLockObjectsNV (HANDLE hDevice, GLint count, HANDLE *hObjects);
        BOOL WINAPI wglDXUnlockObjectsNV (HANDLE hDevice, GLint count, HANDLE *hObjects);
        #endif
        #endif /* WGL_NV_DX_interop */

        #ifndef WGL_NV_DX_interop2
        #define WGL_NV_DX_interop2 1
        #endif /* WGL_NV_DX_interop2 */

        #ifndef WGL_NV_copy_image
        #define WGL_NV_copy_image 1
        typedef BOOL (WINAPI * PFNWGLCOPYIMAGESUBDATANVPROC) (HGLRC hSrcRC, GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, HGLRC hDstRC, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglCopyImageSubDataNV (HGLRC hSrcRC, GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, HGLRC hDstRC, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);
        #endif
        #endif /* WGL_NV_copy_image */

        #ifndef WGL_NV_delay_before_swap
        #define WGL_NV_delay_before_swap 1
        typedef BOOL (WINAPI * PFNWGLDELAYBEFORESWAPNVPROC) (HDC hDC, GLfloat seconds);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglDelayBeforeSwapNV (HDC hDC, GLfloat seconds);
        #endif
        #endif /* WGL_NV_delay_before_swap */

        #ifndef WGL_NV_float_buffer
        #define WGL_NV_float_buffer 1
        #define WGL_FLOAT_COMPONENTS_NV           0x20B0
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV 0x20B1
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV 0x20B2
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV 0x20B3
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV 0x20B4
        #define WGL_TEXTURE_FLOAT_R_NV            0x20B5
        #define WGL_TEXTURE_FLOAT_RG_NV           0x20B6
        #define WGL_TEXTURE_FLOAT_RGB_NV          0x20B7
        #define WGL_TEXTURE_FLOAT_RGBA_NV         0x20B8
        #endif /* WGL_NV_float_buffer */

        #ifndef WGL_NV_gpu_affinity
        #define WGL_NV_gpu_affinity 1
        DECLARE_HANDLE(HGPUNV);
        struct _GPU_DEVICE {
            DWORD  cb;
            CHAR   DeviceName[32];
            CHAR   DeviceString[128];
            DWORD  Flags;
            RECT   rcVirtualScreen;
        };
        typedef struct _GPU_DEVICE *PGPU_DEVICE;
        #define ERROR_INCOMPATIBLE_AFFINITY_MASKS_NV 0x20D0
        #define ERROR_MISSING_AFFINITY_MASK_NV    0x20D1
        typedef BOOL (WINAPI * PFNWGLENUMGPUSNVPROC) (UINT iGpuIndex, HGPUNV *phGpu);
        typedef BOOL (WINAPI * PFNWGLENUMGPUDEVICESNVPROC) (HGPUNV hGpu, UINT iDeviceIndex, PGPU_DEVICE lpGpuDevice);
        typedef HDC (WINAPI * PFNWGLCREATEAFFINITYDCNVPROC) (const HGPUNV *phGpuList);
        typedef BOOL (WINAPI * PFNWGLENUMGPUSFROMAFFINITYDCNVPROC) (HDC hAffinityDC, UINT iGpuIndex, HGPUNV *hGpu);
        typedef BOOL (WINAPI * PFNWGLDELETEDCNVPROC) (HDC hdc);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglEnumGpusNV (UINT iGpuIndex, HGPUNV *phGpu);
        BOOL WINAPI wglEnumGpuDevicesNV (HGPUNV hGpu, UINT iDeviceIndex, PGPU_DEVICE lpGpuDevice);
        HDC WINAPI wglCreateAffinityDCNV (const HGPUNV *phGpuList);
        BOOL WINAPI wglEnumGpusFromAffinityDCNV (HDC hAffinityDC, UINT iGpuIndex, HGPUNV *hGpu);
        BOOL WINAPI wglDeleteDCNV (HDC hdc);
        #endif
        #endif /* WGL_NV_gpu_affinity */

        #ifndef WGL_NV_multigpu_context
        #define WGL_NV_multigpu_context 1
        #define WGL_CONTEXT_MULTIGPU_ATTRIB_NV    0x20AA
        #define WGL_CONTEXT_MULTIGPU_ATTRIB_SINGLE_NV 0x20AB
        #define WGL_CONTEXT_MULTIGPU_ATTRIB_AFR_NV 0x20AC
        #define WGL_CONTEXT_MULTIGPU_ATTRIB_MULTICAST_NV 0x20AD
        #define WGL_CONTEXT_MULTIGPU_ATTRIB_MULTI_DISPLAY_MULTICAST_NV 0x20AE
        #endif /* WGL_NV_multigpu_context */

        #ifndef WGL_NV_multisample_coverage
        #define WGL_NV_multisample_coverage 1
        #define WGL_COVERAGE_SAMPLES_NV           0x2042
        #define WGL_COLOR_SAMPLES_NV              0x20B9
        #endif /* WGL_NV_multisample_coverage */

        #ifndef WGL_NV_present_video
        #define WGL_NV_present_video 1
        DECLARE_HANDLE(HVIDEOOUTPUTDEVICENV);
        #define WGL_NUM_VIDEO_SLOTS_NV            0x20F0
        typedef int (WINAPI * PFNWGLENUMERATEVIDEODEVICESNVPROC) (HDC hDc, HVIDEOOUTPUTDEVICENV *phDeviceList);
        typedef BOOL (WINAPI * PFNWGLBINDVIDEODEVICENVPROC) (HDC hDc, unsigned int uVideoSlot, HVIDEOOUTPUTDEVICENV hVideoDevice, const int *piAttribList);
        typedef BOOL (WINAPI * PFNWGLQUERYCURRENTCONTEXTNVPROC) (int iAttribute, int *piValue);
        #ifdef WGL_WGLEXT_PROTOTYPES
        int WINAPI wglEnumerateVideoDevicesNV (HDC hDc, HVIDEOOUTPUTDEVICENV *phDeviceList);
        BOOL WINAPI wglBindVideoDeviceNV (HDC hDc, unsigned int uVideoSlot, HVIDEOOUTPUTDEVICENV hVideoDevice, const int *piAttribList);
        BOOL WINAPI wglQueryCurrentContextNV (int iAttribute, int *piValue);
        #endif
        #endif /* WGL_NV_present_video */

        #ifndef WGL_NV_render_depth_texture
        #define WGL_NV_render_depth_texture 1
        #define WGL_BIND_TO_TEXTURE_DEPTH_NV      0x20A3
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_DEPTH_NV 0x20A4
        #define WGL_DEPTH_TEXTURE_FORMAT_NV       0x20A5
        #define WGL_TEXTURE_DEPTH_COMPONENT_NV    0x20A6
        #define WGL_DEPTH_COMPONENT_NV            0x20A7
        #endif /* WGL_NV_render_depth_texture */

        #ifndef WGL_NV_render_texture_rectangle
        #define WGL_NV_render_texture_rectangle 1
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV 0x20A0
        #define WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV 0x20A1
        #define WGL_TEXTURE_RECTANGLE_NV          0x20A2
        #endif /* WGL_NV_render_texture_rectangle */

        #ifndef WGL_NV_swap_group
        #define WGL_NV_swap_group 1
        typedef BOOL (WINAPI * PFNWGLJOINSWAPGROUPNVPROC) (HDC hDC, GLuint group);
        typedef BOOL (WINAPI * PFNWGLBINDSWAPBARRIERNVPROC) (GLuint group, GLuint barrier);
        typedef BOOL (WINAPI * PFNWGLQUERYSWAPGROUPNVPROC) (HDC hDC, GLuint *group, GLuint *barrier);
        typedef BOOL (WINAPI * PFNWGLQUERYMAXSWAPGROUPSNVPROC) (HDC hDC, GLuint *maxGroups, GLuint *maxBarriers);
        typedef BOOL (WINAPI * PFNWGLQUERYFRAMECOUNTNVPROC) (HDC hDC, GLuint *count);
        typedef BOOL (WINAPI * PFNWGLRESETFRAMECOUNTNVPROC) (HDC hDC);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglJoinSwapGroupNV (HDC hDC, GLuint group);
        BOOL WINAPI wglBindSwapBarrierNV (GLuint group, GLuint barrier);
        BOOL WINAPI wglQuerySwapGroupNV (HDC hDC, GLuint *group, GLuint *barrier);
        BOOL WINAPI wglQueryMaxSwapGroupsNV (HDC hDC, GLuint *maxGroups, GLuint *maxBarriers);
        BOOL WINAPI wglQueryFrameCountNV (HDC hDC, GLuint *count);
        BOOL WINAPI wglResetFrameCountNV (HDC hDC);
        #endif
        #endif /* WGL_NV_swap_group */

        #ifndef WGL_NV_vertex_array_range
        #define WGL_NV_vertex_array_range 1
        typedef void *(WINAPI * PFNWGLALLOCATEMEMORYNVPROC) (GLsizei size, GLfloat readfreq, GLfloat writefreq, GLfloat priority);
        typedef void (WINAPI * PFNWGLFREEMEMORYNVPROC) (void *pointer);
        #ifdef WGL_WGLEXT_PROTOTYPES
        void *WINAPI wglAllocateMemoryNV (GLsizei size, GLfloat readfreq, GLfloat writefreq, GLfloat priority);
        void WINAPI wglFreeMemoryNV (void *pointer);
        #endif
        #endif /* WGL_NV_vertex_array_range */

        #ifndef WGL_NV_video_capture
        #define WGL_NV_video_capture 1
        DECLARE_HANDLE(HVIDEOINPUTDEVICENV);
        #define WGL_UNIQUE_ID_NV                  0x20CE
        #define WGL_NUM_VIDEO_CAPTURE_SLOTS_NV    0x20CF
        typedef BOOL (WINAPI * PFNWGLBINDVIDEOCAPTUREDEVICENVPROC) (UINT uVideoSlot, HVIDEOINPUTDEVICENV hDevice);
        typedef UINT (WINAPI * PFNWGLENUMERATEVIDEOCAPTUREDEVICESNVPROC) (HDC hDc, HVIDEOINPUTDEVICENV *phDeviceList);
        typedef BOOL (WINAPI * PFNWGLLOCKVIDEOCAPTUREDEVICENVPROC) (HDC hDc, HVIDEOINPUTDEVICENV hDevice);
        typedef BOOL (WINAPI * PFNWGLQUERYVIDEOCAPTUREDEVICENVPROC) (HDC hDc, HVIDEOINPUTDEVICENV hDevice, int iAttribute, int *piValue);
        typedef BOOL (WINAPI * PFNWGLRELEASEVIDEOCAPTUREDEVICENVPROC) (HDC hDc, HVIDEOINPUTDEVICENV hDevice);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglBindVideoCaptureDeviceNV (UINT uVideoSlot, HVIDEOINPUTDEVICENV hDevice);
        UINT WINAPI wglEnumerateVideoCaptureDevicesNV (HDC hDc, HVIDEOINPUTDEVICENV *phDeviceList);
        BOOL WINAPI wglLockVideoCaptureDeviceNV (HDC hDc, HVIDEOINPUTDEVICENV hDevice);
        BOOL WINAPI wglQueryVideoCaptureDeviceNV (HDC hDc, HVIDEOINPUTDEVICENV hDevice, int iAttribute, int *piValue);
        BOOL WINAPI wglReleaseVideoCaptureDeviceNV (HDC hDc, HVIDEOINPUTDEVICENV hDevice);
        #endif
        #endif /* WGL_NV_video_capture */

        #ifndef WGL_NV_video_output
        #define WGL_NV_video_output 1
        DECLARE_HANDLE(HPVIDEODEV);
        #define WGL_BIND_TO_VIDEO_RGB_NV          0x20C0
        #define WGL_BIND_TO_VIDEO_RGBA_NV         0x20C1
        #define WGL_BIND_TO_VIDEO_RGB_AND_DEPTH_NV 0x20C2
        #define WGL_VIDEO_OUT_COLOR_NV            0x20C3
        #define WGL_VIDEO_OUT_ALPHA_NV            0x20C4
        #define WGL_VIDEO_OUT_DEPTH_NV            0x20C5
        #define WGL_VIDEO_OUT_COLOR_AND_ALPHA_NV  0x20C6
        #define WGL_VIDEO_OUT_COLOR_AND_DEPTH_NV  0x20C7
        #define WGL_VIDEO_OUT_FRAME               0x20C8
        #define WGL_VIDEO_OUT_FIELD_1             0x20C9
        #define WGL_VIDEO_OUT_FIELD_2             0x20CA
        #define WGL_VIDEO_OUT_STACKED_FIELDS_1_2  0x20CB
        #define WGL_VIDEO_OUT_STACKED_FIELDS_2_1  0x20CC
        typedef BOOL (WINAPI * PFNWGLGETVIDEODEVICENVPROC) (HDC hDC, int numDevices, HPVIDEODEV *hVideoDevice);
        typedef BOOL (WINAPI * PFNWGLRELEASEVIDEODEVICENVPROC) (HPVIDEODEV hVideoDevice);
        typedef BOOL (WINAPI * PFNWGLBINDVIDEOIMAGENVPROC) (HPVIDEODEV hVideoDevice, HPBUFFERARB hPbuffer, int iVideoBuffer);
        typedef BOOL (WINAPI * PFNWGLRELEASEVIDEOIMAGENVPROC) (HPBUFFERARB hPbuffer, int iVideoBuffer);
        typedef BOOL (WINAPI * PFNWGLSENDPBUFFERTOVIDEONVPROC) (HPBUFFERARB hPbuffer, int iBufferType, unsigned long *pulCounterPbuffer, BOOL bBlock);
        typedef BOOL (WINAPI * PFNWGLGETVIDEOINFONVPROC) (HPVIDEODEV hpVideoDevice, unsigned long *pulCounterOutputPbuffer, unsigned long *pulCounterOutputVideo);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetVideoDeviceNV (HDC hDC, int numDevices, HPVIDEODEV *hVideoDevice);
        BOOL WINAPI wglReleaseVideoDeviceNV (HPVIDEODEV hVideoDevice);
        BOOL WINAPI wglBindVideoImageNV (HPVIDEODEV hVideoDevice, HPBUFFERARB hPbuffer, int iVideoBuffer);
        BOOL WINAPI wglReleaseVideoImageNV (HPBUFFERARB hPbuffer, int iVideoBuffer);
        BOOL WINAPI wglSendPbufferToVideoNV (HPBUFFERARB hPbuffer, int iBufferType, unsigned long *pulCounterPbuffer, BOOL bBlock);
        BOOL WINAPI wglGetVideoInfoNV (HPVIDEODEV hpVideoDevice, unsigned long *pulCounterOutputPbuffer, unsigned long *pulCounterOutputVideo);
        #endif
        #endif /* WGL_NV_video_output */

        #ifndef WGL_OML_sync_control
        #define WGL_OML_sync_control 1
        typedef BOOL (WINAPI * PFNWGLGETSYNCVALUESOMLPROC) (HDC hdc, INT64 *ust, INT64 *msc, INT64 *sbc);
        typedef BOOL (WINAPI * PFNWGLGETMSCRATEOMLPROC) (HDC hdc, INT32 *numerator, INT32 *denominator);
        typedef INT64 (WINAPI * PFNWGLSWAPBUFFERSMSCOMLPROC) (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder);
        typedef INT64 (WINAPI * PFNWGLSWAPLAYERBUFFERSMSCOMLPROC) (HDC hdc, INT fuPlanes, INT64 target_msc, INT64 divisor, INT64 remainder);
        typedef BOOL (WINAPI * PFNWGLWAITFORMSCOMLPROC) (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder, INT64 *ust, INT64 *msc, INT64 *sbc);
        typedef BOOL (WINAPI * PFNWGLWAITFORSBCOMLPROC) (HDC hdc, INT64 target_sbc, INT64 *ust, INT64 *msc, INT64 *sbc);
        #ifdef WGL_WGLEXT_PROTOTYPES
        BOOL WINAPI wglGetSyncValuesOML (HDC hdc, INT64 *ust, INT64 *msc, INT64 *sbc);
        BOOL WINAPI wglGetMscRateOML (HDC hdc, INT32 *numerator, INT32 *denominator);
        INT64 WINAPI wglSwapBuffersMscOML (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder);
        INT64 WINAPI wglSwapLayerBuffersMscOML (HDC hdc, INT fuPlanes, INT64 target_msc, INT64 divisor, INT64 remainder);
        BOOL WINAPI wglWaitForMscOML (HDC hdc, INT64 target_msc, INT64 divisor, INT64 remainder, INT64 *ust, INT64 *msc, INT64 *sbc);
        BOOL WINAPI wglWaitForSbcOML (HDC hdc, INT64 target_sbc, INT64 *ust, INT64 *msc, INT64 *sbc);
        #endif
        #endif /* WGL_OML_sync_control */

        #ifdef __cplusplus
        }
      #endif
    }

    #elif defined(fan_platform_unix)
    
    namespace glx {
      #define GLX_ACCUM_ALPHA_SIZE 17
      #define GLX_ACCUM_BLUE_SIZE 16
      #define GLX_ACCUM_BUFFER_BIT 0x00000080
      #define GLX_ACCUM_GREEN_SIZE 15
      #define GLX_ACCUM_RED_SIZE 14
      #define GLX_ALPHA_SIZE 11
      #define GLX_AUX_BUFFERS 7
      #define GLX_AUX_BUFFERS_BIT 0x00000010
      #define GLX_BACK_LEFT_BUFFER_BIT 0x00000004
      #define GLX_BACK_RIGHT_BUFFER_BIT 0x00000008
      #define GLX_BAD_ATTRIBUTE 2
      #define GLX_BAD_CONTEXT 5
      #define GLX_BAD_ENUM 7
      #define GLX_BAD_SCREEN 1
      #define GLX_BAD_VALUE 6
      #define GLX_BAD_VISUAL 4
      #define GLX_BLUE_SIZE 10
      #define GLX_BUFFER_SIZE 2
      #define GLX_BufferSwapComplete 1
      #define GLX_COLOR_INDEX_BIT 0x00000002
      #define GLX_COLOR_INDEX_TYPE 0x8015
      #define GLX_CONFIG_CAVEAT 0x20
      #define GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
      #define GLX_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
      #define GLX_CONTEXT_DEBUG_BIT_ARB 0x00000001
      #define GLX_CONTEXT_FLAGS_ARB 0x2094
      #define GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
      #define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
      #define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092
      #define GLX_CONTEXT_PROFILE_MASK_ARB 0x9126
      #define GLX_DAMAGED 0x8020
      #define GLX_DEPTH_BUFFER_BIT 0x00000020
      #define GLX_DEPTH_SIZE 12
      #define GLX_DIRECT_COLOR 0x8003
      #define GLX_DONT_CARE 0xFFFFFFFF
      #define GLX_DOUBLEBUFFER 5
      #define GLX_DRAWABLE_TYPE 0x8010
      #define GLX_EVENT_MASK 0x801F
      #define GLX_EXTENSIONS 0x3
      #define GLX_EXTENSION_NAME "GLX"
      #define GLX_FBCONFIG_ID 0x8013
      #define GLX_FRONT_LEFT_BUFFER_BIT 0x00000001
      #define GLX_FRONT_RIGHT_BUFFER_BIT 0x00000002
      #define GLX_GRAY_SCALE 0x8006
      #define GLX_GREEN_SIZE 9
      #define GLX_HEIGHT 0x801E
      #define GLX_LARGEST_PBUFFER 0x801C
      #define GLX_LEVEL 3
      #define GLX_MAX_PBUFFER_HEIGHT 0x8017
      #define GLX_MAX_PBUFFER_PIXELS 0x8018
      #define GLX_MAX_PBUFFER_WIDTH 0x8016
      #define GLX_MAX_SWAP_INTERVAL_EXT 0x20F2
      #define GLX_NONE 0x8000
      #define GLX_NON_CONFORMANT_CONFIG 0x800D
      #define GLX_NO_EXTENSION 3
      #define GLX_PBUFFER 0x8023
      #define GLX_PBUFFER_BIT 0x00000004
      #define GLX_PBUFFER_CLOBBER_MASK 0x08000000
      #define GLX_PBUFFER_HEIGHT 0x8040
      #define GLX_PBUFFER_WIDTH 0x8041
      #define GLX_PIXMAP_BIT 0x00000002
      #define GLX_PRESERVED_CONTENTS 0x801B
      #define GLX_PSEUDO_COLOR 0x8004
      #define GLX_PbufferClobber 0
      #define GLX_RED_SIZE 8
      #define GLX_RENDER_TYPE 0x8011
      #define GLX_RGBA 4
      #define GLX_RGBA_BIT 0x00000001
      #define GLX_RGBA_TYPE 0x8014
      #define GLX_SAMPLES 100001
      #define GLX_SAMPLE_BUFFERS 100000
      #define GLX_SAVED 0x8021
      #define GLX_SCREEN 0x800C
      #define GLX_SLOW_CONFIG 0x8001
      #define GLX_STATIC_COLOR 0x8005
      #define GLX_STATIC_GRAY 0x8007
      #define GLX_STENCIL_BUFFER_BIT 0x00000040
      #define GLX_STENCIL_SIZE 13
      #define GLX_STEREO 6
      #define GLX_SWAP_INTERVAL_EXT 0x20F1
      #define GLX_TRANSPARENT_ALPHA_VALUE 0x28
      #define GLX_TRANSPARENT_BLUE_VALUE 0x27
      #define GLX_TRANSPARENT_GREEN_VALUE 0x26
      #define GLX_TRANSPARENT_INDEX 0x8009
      #define GLX_TRANSPARENT_INDEX_VALUE 0x24
      #define GLX_TRANSPARENT_RED_VALUE 0x25
      #define GLX_TRANSPARENT_RGB 0x8008
      #define GLX_TRANSPARENT_TYPE 0x23
      #define GLX_TRUE_COLOR 0x8002
      #define GLX_USE_GL 1
      #define GLX_VENDOR 0x1
      #define GLX_VERSION 0x2
      #define GLX_VISUAL_ID 0x800B
      #define GLX_WIDTH 0x801D
      #define GLX_WINDOW 0x8022
      #define GLX_WINDOW_BIT 0x00000001
      #define GLX_X_RENDERABLE 0x8012
      #define GLX_X_VISUAL_TYPE 0x22
      #define __GLX_NUMBER_EVENTS 17
      typedef XID GLXFBConfigID;
      typedef struct __GLXFBConfigRec* GLXFBConfig;
      typedef XID GLXContextID;
      typedef struct __GLXcontextRec* GLXContext;
      typedef XID GLXPixmap;
      typedef XID GLXDrawable;
      typedef XID GLXWindow;
      typedef XID GLXPbuffer;
      typedef void (FAN_API_PTR* __GLXextFuncPtr)(void);
      typedef XID GLXVideoCaptureDeviceNV;
      typedef unsigned int GLXVideoDeviceNV;
      typedef XID GLXVideoSourceSGIX;
      typedef XID GLXFBConfigIDSGIX;
      typedef struct __GLXFBConfigRec* GLXFBConfigSGIX;
      typedef XID GLXPbufferSGIX;
      typedef struct {
        int event_type;             /* GLX_DAMAGED or GLX_SAVED */
        int draw_type;              /* GLX_WINDOW or GLX_PBUFFER */
        unsigned long serial;       /* # of last request processed by server */
        Bool send_event;            /* true if this came for SendEvent request */
        Display* display;           /* display the event was read from */
        GLXDrawable drawable;       /* XID of Drawable */
        unsigned int buffer_mask;   /* mask indicating which buffers are affected */
        unsigned int aux_buffer;    /* which aux buffer was affected */
        int x, y;
        int width, height;
        int count;                  /* if nonzero, at least this many more */
      } GLXPbufferClobberEvent;
      typedef struct {
        int type;
        unsigned long serial;       /* # of last request processed by server */
        Bool send_event;            /* true if this came from a SendEvent request */
        Display* display;           /* Display the event was read from */
        GLXDrawable drawable;       /* drawable on which event was requested in event mask */
        int event_type;
        int64_t ust;
        int64_t msc;
        int64_t sbc;
      } GLXBufferSwapComplete;
      typedef union __GLXEvent {
        GLXPbufferClobberEvent glxpbufferclobber;
        GLXBufferSwapComplete glxbufferswapcomplete;
        long pad[24];
      } GLXEvent;
      typedef struct {
        int type;
        unsigned long serial;
        Bool send_event;
        Display* display;
        int extension;
        int evtype;
        GLXDrawable window;
        Bool stereo_tree;
      } GLXStereoNotifyEventEXT;
      typedef struct {
        int type;
        unsigned long serial;   /* # of last request processed by server */
        Bool send_event;        /* true if this came for SendEvent request */
        Display* display;       /* display the event was read from */
        GLXDrawable drawable;   /* i.d. of Drawable */
        int event_type;         /* GLX_DAMAGED_SGIX or GLX_SAVED_SGIX */
        int draw_type;          /* GLX_WINDOW_SGIX or GLX_PBUFFER_SGIX */
        unsigned int mask;      /* mask indicating which buffers are affected*/
        int x, y;
        int width, height;
        int count;              /* if nonzero, at least this many more */
      } GLXBufferClobberEventSGIX;
      typedef struct {
        char    pipeName[80]; /* Should be [GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX] */
        int     networkId;
      } GLXHyperpipeNetworkSGIX;
      typedef struct {
        char    pipeName[80]; /* Should be [GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX] */
        int     channel;
        unsigned int participationType;
        int     timeSlice;
      } GLXHyperpipeConfigSGIX;
      typedef struct {
        char pipeName[80]; /* Should be [GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX] */
        int srcXOrigin, srcYOrigin, srcWidth, srcHeight;
        int destXOrigin, destYOrigin, destWidth, destHeight;
      } GLXPipeRect;
      typedef struct {
        char pipeName[80]; /* Should be [GLX_HYPERPIPE_PIPE_NAME_LENGTH_SGIX] */
        int XOrigin, YOrigin, maxHeight, maxWidth;
      } GLXPipeRectLimits;

      typedef GLXFBConfig* (FAN_API_PTR* PFNGLXCHOOSEFBCONFIGPROC)(Display* dpy, int screen, const int* attrib_list, int* nelements);
      typedef XVisualInfo* (FAN_API_PTR* PFNGLXCHOOSEVISUALPROC)(Display* dpy, int screen, int* attribList);
      typedef void (FAN_API_PTR* PFNGLXCOPYCONTEXTPROC)(Display* dpy, GLXContext src, GLXContext dst, unsigned long mask);
      typedef GLXContext(FAN_API_PTR* PFNGLXCREATECONTEXTPROC)(Display* dpy, XVisualInfo* vis, GLXContext shareList, Bool direct);
      typedef GLXContext(FAN_API_PTR* PFNGLXCREATECONTEXTATTRIBSARBPROC)(Display* dpy, GLXFBConfig config, GLXContext share_context, Bool direct, const int* attrib_list);
      typedef GLXPixmap(FAN_API_PTR* PFNGLXCREATEGLXPIXMAPPROC)(Display* dpy, XVisualInfo* visual, Pixmap pixmap);
      typedef GLXContext(FAN_API_PTR* PFNGLXCREATENEWCONTEXTPROC)(Display* dpy, GLXFBConfig config, int render_type, GLXContext share_list, Bool direct);
      typedef GLXPbuffer(FAN_API_PTR* PFNGLXCREATEPBUFFERPROC)(Display* dpy, GLXFBConfig config, const int* attrib_list);
      typedef GLXPixmap(FAN_API_PTR* PFNGLXCREATEPIXMAPPROC)(Display* dpy, GLXFBConfig config, Pixmap pixmap, const int* attrib_list);
      typedef GLXWindow(FAN_API_PTR* PFNGLXCREATEWINDOWPROC)(Display* dpy, GLXFBConfig config, Window win, const int* attrib_list);
      typedef void (FAN_API_PTR* PFNGLXDESTROYCONTEXTPROC)(Display* dpy, GLXContext ctx);
      typedef void (FAN_API_PTR* PFNGLXDESTROYGLXPIXMAPPROC)(Display* dpy, GLXPixmap pixmap);
      typedef void (FAN_API_PTR* PFNGLXDESTROYPBUFFERPROC)(Display* dpy, GLXPbuffer pbuf);
      typedef void (FAN_API_PTR* PFNGLXDESTROYPIXMAPPROC)(Display* dpy, GLXPixmap pixmap);
      typedef void (FAN_API_PTR* PFNGLXDESTROYWINDOWPROC)(Display* dpy, GLXWindow win);
      typedef const char* (FAN_API_PTR* PFNGLXGETCLIENTSTRINGPROC)(Display* dpy, int name);
      typedef int (FAN_API_PTR* PFNGLXGETCONFIGPROC)(Display* dpy, XVisualInfo* visual, int attrib, int* value);
      typedef GLXContext(FAN_API_PTR* PFNGLXGETCURRENTCONTEXTPROC)(void);
      typedef Display* (FAN_API_PTR* PFNGLXGETCURRENTDISPLAYPROC)(void);
      typedef GLXDrawable(FAN_API_PTR* PFNGLXGETCURRENTDRAWABLEPROC)(void);
      typedef GLXDrawable(FAN_API_PTR* PFNGLXGETCURRENTREADDRAWABLEPROC)(void);
      typedef int (FAN_API_PTR* PFNGLXGETFBCONFIGATTRIBPROC)(Display* dpy, GLXFBConfig config, int attribute, int* value);
      typedef GLXFBConfig* (FAN_API_PTR* PFNGLXGETFBCONFIGSPROC)(Display* dpy, int screen, int* nelements);
      typedef __GLXextFuncPtr(FAN_API_PTR* PFNGLXGETPROCADDRESSPROC)(const GLubyte* procName);
      typedef __GLXextFuncPtr(FAN_API_PTR* PFNGLXGETPROCADDRESSARBPROC)(const GLubyte* procName);
      typedef void (FAN_API_PTR* PFNGLXGETSELECTEDEVENTPROC)(Display* dpy, GLXDrawable draw, unsigned long* event_mask);
      typedef XVisualInfo* (FAN_API_PTR* PFNGLXGETVISUALFROMFBCONFIGPROC)(Display* dpy, GLXFBConfig config);
      typedef Bool(FAN_API_PTR* PFNGLXISDIRECTPROC)(Display* dpy, GLXContext ctx);
      typedef Bool(FAN_API_PTR* PFNGLXMAKECONTEXTCURRENTPROC)(Display* dpy, GLXDrawable draw, GLXDrawable read, GLXContext ctx);
      typedef Bool(FAN_API_PTR* PFNGLXMAKECURRENTPROC)(Display* dpy, GLXDrawable drawable, GLXContext ctx);
      typedef int (FAN_API_PTR* PFNGLXQUERYCONTEXTPROC)(Display* dpy, GLXContext ctx, int attribute, int* value);
      typedef void (FAN_API_PTR* PFNGLXQUERYDRAWABLEPROC)(Display* dpy, GLXDrawable draw, int attribute, unsigned int* value);
      typedef Bool(FAN_API_PTR* PFNGLXQUERYEXTENSIONPROC)(Display* dpy, int* errorb, int* event);
      typedef const char* (FAN_API_PTR* PFNGLXQUERYEXTENSIONSSTRINGPROC)(Display* dpy, int screen);
      typedef const char* (FAN_API_PTR* PFNGLXQUERYSERVERSTRINGPROC)(Display* dpy, int screen, int name);
      typedef Bool(FAN_API_PTR* PFNGLXQUERYVERSIONPROC)(Display* dpy, int* maj, int* min);
      typedef void (FAN_API_PTR* PFNGLXSELECTEVENTPROC)(Display* dpy, GLXDrawable draw, unsigned long event_mask);
      typedef void (FAN_API_PTR* PFNGLXSWAPBUFFERSPROC)(Display* dpy, GLXDrawable drawable);
      typedef void (FAN_API_PTR* PFNGLXSWAPINTERVALEXTPROC)(Display* dpy, GLXDrawable drawable, int interval);
      typedef void (FAN_API_PTR* PFNGLXUSEXFONTPROC)(Font font, int first, int count, int list);
      typedef void (FAN_API_PTR* PFNGLXWAITGLPROC)(void);
      typedef void (FAN_API_PTR* PFNGLXWAITXPROC)(void);
    }

    #endif

  }
}

// X11

//#ifdef DestroyAll
//#undef DestroyAll
//#endif
//
//#ifdef None
//#undef None
//#endif
//
//#ifdef Bool
//#undef Bool
//#endif
//
//#ifdef True
//#undef True
//#endif
//
//#ifdef False
//#undef False
//#endif
//
//#ifdef Status
//#undef Status
//#endif
//
//#ifdef Success
//#undef Success
//#endif
//
//#ifdef Complex
//#undef Complex
//#endif
//
//#ifdef Cursor
//#undef Cursor
//#endif
//
//#ifdef Gravity
//#undef Gravity
//#endif
//
//#ifdef Mask
//#undef Mask
//#endif
//
//#ifdef Window
//#undef Window
//#endif
//
//#ifdef Button
//#undef Button
//#endif
//
//#ifdef Above
//#undef Above
//#endif
//
//#ifdef Below
//#undef Below
//#endif
//
//#ifdef FocusIn
//#undef FocusIn
//#endif
//
//#ifdef FocusOut
//#undef FocusOut
//#endif
//
//#ifdef FontChange
//#undef FontChange
//#endif
//
//#ifdef KeyPress
//#undef KeyPress
//#endif
//
//#ifdef KeyRelease
//#undef KeyRelease
//#endif
//
//#ifdef Create
//#undef Create
//#endif
//
//#ifdef Expose
//#undef Expose
//#endif
//
//#ifdef Destroy
//#undef Destroy
//#endif
//
//#ifdef Always
//#undef Always
//#endif