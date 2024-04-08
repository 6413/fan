#if defined(loco_sprite_sheet)
#define loco_sprite
#endif
#if defined(loco_sprite)
#if defined(loco_opengl)
#define loco_texture_pack
#define loco_unlit_sprite
#endif
#endif

#if defined(loco_button)
#define loco_letter
#define loco_text
#define loco_vfi
#endif

#if defined(loco_text)
#define loco_letter
#define loco_responsive_text
#endif

#ifdef loco_vulkan
#ifdef loco_line 
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#endif
#ifdef loco_rectangle 
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#endif
#ifdef loco_sprite
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#ifndef loco_vulkan_descriptor_image_sampler
#define loco_vulkan_descriptor_image_sampler
#endif
#endif
#ifdef loco_yuv420p
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#ifndef loco_vulkan_descriptor_image_sampler
#define loco_vulkan_descriptor_image_sampler
#endif
#endif
#ifdef loco_letter
#ifndef loco_vulkan_descriptor_ssbo
#define loco_vulkan_descriptor_ssbo
#endif
#ifndef loco_vulkan_descriptor_uniform_block
#define loco_vulkan_descriptor_uniform_block
#endif
#ifndef loco_vulkan_descriptor_image_sampler
#define loco_vulkan_descriptor_image_sampler
#endif
#endif
#if defined loco_compute_shader
#define loco_vulkan_descriptor_ssbo
#endif
#endif