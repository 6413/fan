struct frame_capture_t {

	loco_t* get_loco() {
		loco_t* loco = OFFSETLESS(this, loco_t, sb_frame_capture_var_name);
		return loco;
	}

	struct bloom_t {

		struct mip_t {
			fan::vec2 size;
			fan::vec2i int_size;
			fan::opengl::image_t image;
		};

		frame_capture_t* get_frame_capture() {
			return OFFSETLESS(this, frame_capture_t, capture);
		}

		loco_t* get_loco() {
			return get_frame_capture()->get_loco();
		}

		// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
		unsigned int quadVAO = 0;
		unsigned int quadVBO;
		void renderQuad()
		{
			auto loco = get_loco();
			if (quadVAO == 0)
			{
				float quadVertices[] = {
					// positions        // texture Coords
					-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
					-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
					 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
					 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
				};
				// setup plane VAO
				loco->get_context().opengl.glGenVertexArrays(1, &quadVAO);
				loco->get_context().opengl.glGenBuffers(1, &quadVBO);
				loco->get_context().opengl.glBindVertexArray(quadVAO);
				loco->get_context().opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, quadVBO);
				loco->get_context().opengl.glBufferData(fan::opengl::GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, fan::opengl::GL_STATIC_DRAW);
				loco->get_context().opengl.glEnableVertexAttribArray(0);
				loco->get_context().opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)0);
				loco->get_context().opengl.glEnableVertexAttribArray(1);
				loco->get_context().opengl.glVertexAttribPointer(1, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
			}
			loco->get_context().opengl.glBindVertexArray(quadVAO);
			loco->get_context().opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
			loco->get_context().opengl.glBindVertexArray(0);
		}

		void open(const fan::vec2& resolution, uint32_t mip_count) {

			auto loco = get_loco();

			framebuffer.open(loco->get_context());
			framebuffer.bind(loco->get_context());

			shader_downsample.open(loco->get_context());
			shader_downsample.set_vertex(
				loco->get_context(),
				#include _FAN_PATH(graphics/glsl/opengl/2D/effects/downsample.vs)
			);
			shader_downsample.set_fragment(
				loco->get_context(),
				#include _FAN_PATH(graphics/glsl/opengl/2D/objects/rectangle.fs)
			);
			shader_downsample.compile(loco->get_context());

			shader_downsample.use(loco->get_context());
			shader_downsample.set_int(loco->get_context(), "_t00", 0);

			shader_bloom.open(loco->get_context());
			shader_bloom.set_vertex(
				loco->get_context(),
				#include _FAN_PATH(graphics/glsl/opengl/2D/effects/downsample.vs)
			);
			shader_bloom.set_fragment(
				loco->get_context(),
				#include _FAN_PATH(graphics/glsl/opengl/2D/effects/frame_capture.fs)
			);
			shader_bloom.compile(loco->get_context());

			fan::vec2 mip_size = resolution;
			fan::vec2i mip_int_size = resolution;

			for (uint32_t i = 0; i < mip_count; i++) {
				mip_t mip;

				mip_size *= 0.5;
				mip_int_size /= 2;
				mip.size = mip_size;
				mip.int_size = mip_int_size;

				fan::opengl::image_t::load_properties_t lp;
				lp.internal_format = fan::opengl::GL_R11F_G11F_B10F;
				lp.format = fan::opengl::GL_RGB;
				lp.type = fan::opengl::GL_FLOAT;
				lp.filter = fan::opengl::GL_LINEAR;
				lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;
				fan::webp::image_info_t ii;
				ii.data = nullptr;
				ii.size = mip_size;
				mip.image.load(loco->get_context(), ii, lp);
				mips.push_back(mip);
			}

			loco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
			mips[0].image.bind_texture(loco->get_context());
			framebuffer.bind_to_texture(
				loco->get_context(),
				*mips[0].image.get_texture(loco->get_context()),
				fan::opengl::GL_COLOR_ATTACHMENT0
			);

			if (!framebuffer.ready(loco->get_context())) {
				fan::throw_error("framebuffer not ready");
			}

			unsigned int attachments[1] = { fan::opengl::GL_COLOR_ATTACHMENT0 };
			loco->get_context().opengl.glDrawBuffers(1, attachments);

			framebuffer.unbind(loco->get_context());
		}
			
		void draw_downsamples(fan::opengl::image_t* image) {
			auto loco = get_loco();
			loco->get_context().set_depth_test(false);

			auto pp = get_frame_capture();
		
			fan::vec2 window_size = loco->window.get_size();
			
			shader_downsample.use(loco->get_context());
			shader_downsample.set_vec2(loco->get_context(), "resolution", window_size);
			
			loco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
			image->bind_texture(loco->get_context());

			for (uint32_t i = 0; i < mips.size(); i++) {
				mip_t mip = mips[i];
				loco->get_context().opengl.glViewport(0, 0, mip.size.x, mip.size.y);
				framebuffer.bind_to_texture(
					loco->get_context(), 
					*mip.image.get_texture(loco->get_context()), 
					fan::opengl::GL_COLOR_ATTACHMENT0
				);

				renderQuad();

				mip.image.bind_texture(loco->get_context());
			}
		}

		void draw(fan::opengl::image_t* color_texture, f32_t filter_radius) {
			auto loco = get_loco();

			framebuffer.bind(loco->get_context());

			draw_downsamples(color_texture);
			//draw_upsamples(filter_radius);

			framebuffer.unbind(loco->get_context());
			
			fan::vec2 window_size = loco->window.get_size();
			loco->get_context().opengl.glViewport(0, 0, window_size.x, window_size.y);
		}

		std::vector<mip_t> mips;

		fan::opengl::core::renderbuffer_t renderbuffer;
		fan::opengl::core::framebuffer_t framebuffer;

		fan::opengl::shader_t shader_downsample;
		fan::opengl::shader_t shader_upsample;
		fan::opengl::shader_t shader_bloom;
	}capture;

	bool open(const fan::opengl::core::renderbuffer_t::properties_t& p) {
		auto loco = get_loco();

		fan::opengl::image_t::load_properties_t lp;
		lp.internal_format = fan::opengl::GL_RGBA16F;
		lp.format = fan::opengl::GL_RGBA;
		lp.type = fan::opengl::GL_FLOAT;
		lp.filter = fan::opengl::GL_LINEAR;
		lp.visual_output = fan::opengl::GL_CLAMP_TO_EDGE;

		hdr_fbo.open(loco->get_context());
		hdr_fbo.bind(loco->get_context());

		fan::webp::image_info_t ii;
		ii.data = nullptr;
		ii.size = loco->window.get_size();

		color_buffers[0].load(loco->get_context(), ii, lp);

		color_buffers[0].bind_texture(loco->get_context());
		fan::opengl::core::framebuffer_t::bind_to_texture(
			loco->get_context(),
			*color_buffers[0].get_texture(loco->get_context()),
			fan::opengl::GL_COLOR_ATTACHMENT0
		);

		color_buffers[1].load(loco->get_context(), ii, lp);

		color_buffers[1].bind_texture(loco->get_context());
		fan::opengl::core::framebuffer_t::bind_to_texture(
			loco->get_context(),
			*color_buffers[1].get_texture(loco->get_context()),
			fan::opengl::GL_COLOR_ATTACHMENT1
		);

		fan::opengl::core::renderbuffer_t::properties_t rp;
		rp.size = ii.size;
		rp.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
		rbo.open(loco->get_context());
		rbo.set_storage(loco->get_context(), rp);
		rp.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
		rbo.bind_to_renderbuffer(loco->get_context(), rp);

		// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering
		unsigned int attachments[2] = {
			fan::opengl::GL_COLOR_ATTACHMENT0,
			fan::opengl::GL_COLOR_ATTACHMENT1
		};

		loco->get_context().opengl.call(loco->get_context().opengl.glDrawBuffers, 2, attachments);
		// finally check if framebuffer is complete
		if (!hdr_fbo.ready(loco->get_context())) {
			fan::throw_error("framebuffer not ready");
		}

		hdr_fbo.unbind(loco->get_context());

		static constexpr uint32_t mip_count = 1;
		capture.open(loco->window.get_size(), mip_count);

		return 0;
	}
	void close() {
		auto loco = get_loco();

	}

	void push(fan::opengl::viewport_t* viewport, fan::opengl::camera_t* camera) {

		//for (uint32_t i = 0; i < 1; i++) {
		//	post_sprite_t::properties_t sp;
		//	sp.viewport = viewport;
		//	sp.camera = camera;
		//	sp.image = &capture.mips[0].image;
		//	sp.position =  gloco->window.get_size() / 2;
		//	sp.size = gloco->window.get_size() / 2;
		//	sprite.push_back(&cid, sp);
		//}
	}

	//void update_renderbuffer( const fan::opengl::core::renderbuffer_t::properties_t& p) {
	//  auto loco = get_loco();
	//  renderbuffer.set_storage(loco->get_context(), p);
	//}

	void start_capture() {
		auto loco = get_loco();
		loco->get_context().opengl.call(loco->get_context().opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
		hdr_fbo.bind(loco->get_context());
	}
	void end_capture() {
		auto loco = get_loco();
		hdr_fbo.unbind(loco->get_context());
	}

	void draw() {
		end_capture();

		auto loco = get_loco();

		constexpr float bloom_filter_radius = 0.005f;
		capture.draw(&color_buffers[1], bloom_filter_radius);

		loco->get_context().opengl.call(loco->get_context().opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);

		capture.shader_bloom.use(loco->get_context());
		capture.shader_bloom.set_int(loco->get_context(), "_t00", 0);
		capture.shader_bloom.set_int(loco->get_context(), "_t01", 1);
		capture.shader_bloom.set_float(loco->get_context(), "capture", bloomamount);

		loco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
		color_buffers[0].bind_texture(loco->get_context());

		loco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
		capture.mips[0].image.bind_texture(loco->get_context());

		capture.renderQuad();
		//loco->get_context().set_depth_test(true);
		//sprite.m_shader = old_shader;

		//start_capture();
	}

	fan::opengl::core::framebuffer_t hdr_fbo;
	fan::opengl::core::renderbuffer_t rbo;

	fan::opengl::image_t color_buffers[2];

	uint32_t draw_nodereference;

	fan::opengl::cid_t cid;

	f32_t bloomamount = 0.04;
};