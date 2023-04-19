struct model_t {

	struct textures_t {
		fan::opengl::image_t diffuse;
		fan::opengl::image_t depth;
	};

	struct loaded_model_t {
		std::vector<fan::vec3> vertices;
		//fan::hector_t<uint32_t> indices;
		//fan::hector_t<fan::vec3> normals;
		std::vector<fan::vec2> texture_coordinates;
		//fan::hector_t<fan::vec3> tangets;
		textures_t m_textures;
	};

	struct model_instance_t {
		fan::vec3 position = 1;
		fan::vec3 size = 1;
		f32_t angle = 0;
		fan::vec3 rotation_point = 0;
		fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
		uint32_t m_ebo;
		fan::hector_t<uint32_t> m_indices;
		uint32_t is_light;
		fan::vec3 light_position = 0;
	};

	#define hardcode0_t fan::opengl::textureid_t<0>
  #define hardcode0_n image
  #define hardcode1_t fan::opengl::camera_list_NodeReference_t
  #define hardcode1_n camera
  #define hardcode2_t fan::opengl::viewport_list_NodeReference_t
  #define hardcode2_n viewport
  #include _FAN_PATH(graphics/opengl/2D/objects/hardcode_open.h)

	loaded_model_t* load_model(const fan::string& path) {

		fastObjMesh* mesh = fast_obj_read(path.c_str());

		if (mesh == nullptr)
		{
			fan::throw_error("error loading model \"" + path + "\"");
		}

		loaded_model_t* model = new loaded_model_t;

		parse_model(model, mesh);
		// aiMaterial* material = scene->mMaterials[scene->mMeshes[0]->mMaterialIndex];
		model->m_textures = parse_model_texture(mesh);

		return model;
	}

	struct instance_t {
		fan::vec4 position[3];
		fan::vec2 texture_coordinate[3];
		fan::vec2 pad;
		//fan::vec3 normal;
		//fan::vec3 tanget;
	};

	struct instance_properties_t {
		struct key_t : parsed_masterpiece_t {}key;
		expand_get_functions
	};

	struct properties_t : instance_t, instance_properties_t {
		fan::vec3 position;
		fan::vec3 size = 1;
		f32_t angle = 0;
		fan::vec3 rotation_point = 0;
		fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
		loaded_model_t* loaded_model;
		uint32_t is_light = false;
		fan::vec3* camera_position;
	};

	static constexpr uint32_t max_instance_size = fan::min(256, 4096 / (sizeof(instance_t) / 4));
	#define sb_vertex_count 3
	#define sb_shader_vertex_path _FAN_PATH(graphics/glsl/opengl/3D/objects/model.vs)
	#define sb_shader_fragment_path _FAN_PATH(graphics/glsl/opengl/3D/objects/model.fs)
	#include _FAN_PATH(graphics/opengl/2D/objects/shape_builder.h)

	void open() {
		m_model_instance.m_indices.open();
		sb_open();
	}
	void close() {

		sb_close();

		gloco->get_context()->opengl.glDeleteBuffers(1, &m_model_instance.m_ebo);
		m_model_instance.m_indices.close();
	}

	void set(properties_t properties) {
		m_model_instance.position = properties.position;
		m_model_instance.size = properties.size;
		m_model_instance.angle = properties.angle;
		m_model_instance.rotation_point = properties.rotation_point;
		m_model_instance.rotation_vector = properties.rotation_vector;
		m_model_instance.is_light = properties.is_light;
		//m_model_instance.skybox = properties.skybox;
		camera_position = properties.camera_position;
		properties.get_image() = &properties.loaded_model->m_textures.diffuse;

		instance_t instance;
		for (int i = 0; i < properties.loaded_model->vertices.size(); i+=3) {
			*(fan::vec3*)&instance.position[0] = properties.loaded_model->vertices[i + 0];
			*(fan::vec3*)&instance.position[1] = properties.loaded_model->vertices[i + 1];
			*(fan::vec3*)&instance.position[2] = properties.loaded_model->vertices[i + 2];

			*(fan::vec2*)&instance.texture_coordinate[0] = properties.loaded_model->texture_coordinates[i + 0];
			*(fan::vec2*)&instance.texture_coordinate[1] = properties.loaded_model->texture_coordinates[i + 1];
			*(fan::vec2*)&instance.texture_coordinate[2] = properties.loaded_model->texture_coordinates[i + 2];

			fan::opengl::cid_t cid;
			*(instance_t*)&properties = instance;
			sb_push_back(&cid, properties);
		}
	}

	fan::vec3 get_position() const {
		return m_model_instance.position;
	}
	void set_position(const fan::vec3& position) {
		m_model_instance.position = position;
	}

	void set_light_position(const fan::vec3& position) {
		m_model_instance.light_position = position;
	}

	fan::vec3 get_size() const {
		return m_model_instance.size;
	}
	void set_size(const fan::vec3& size) {
		m_model_instance.size = size;
	}

	f32_t get_angle() const {
		return m_model_instance.angle;
	}
	void set_angle(f32_t angle) {
		m_model_instance.angle = angle;
	}

	fan::vec3 get_rotation_vector(uint32_t i) const;
	void set_rotation_vector(uint32_t i, const fan::vec3& rotation_vector) {
		m_model_instance.rotation_vector = rotation_vector;
	}

	void draw() {
		auto context = gloco->get_context();

		context->set_depth_test(true);

		//m_shader.v

		m_shader.use(context);

		fan::mat4 m[2];
		std::memset(m, 0, sizeof(m));
		m[0][0][0] = m_model_instance.position.x;
		m[0][0][1] = m_model_instance.position.y;
		m[0][0][2] = m_model_instance.position.z;
		m[0][0][3] = m_model_instance.size.x;

		m[0][1][0] = m_model_instance.size.y;
		m[0][1][1] = m_model_instance.size.z;
		m[0][1][2] = m_model_instance.angle;
		m[0][1][3] = m_model_instance.rotation_point.x;

		m[0][2][0] = m_model_instance.rotation_point.y;
		m[0][2][1] = m_model_instance.rotation_point.z;
		m[0][2][2] = m_model_instance.rotation_vector.x;
		m[0][2][3] = m_model_instance.rotation_vector.y;

		m[0][3][0] = m_model_instance.rotation_vector.z;
		m[0][3][1] = m_model_instance.is_light;
		// camera position
		m[0][3][2] = camera_position->x;
		m[0][3][3] = camera_position->y;
		m[1][0][0] = camera_position->z;

		/*#if fan_debug >= fan_debug_low
			if (m_textures.diffuse.texture == 0) {
				fan::throw_error("invalid diffuse texture");
			}
			if (m_textures.depth.texture == 0) {
				fan::throw_error("invalid depth texture");
			}
		#endif
		*/
		/*  m_shader.set_int(context, "texture_diffuse", 0);
			context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
			context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_textures.diffuse.texture);*/
			/*
			m_shader.set_int(context, "texture_depth", 1);
			context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 1);
			context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_textures.depth.texture);*/

			/* m_shader.set_int(context, "skybox", 2);
			 context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 2);
			 context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, m_model_instance.skybox.texture);*/

		m_shader.set_mat4(context, "model_input", &m[0][0][0], std::size(m));
		// m_shader.set_float(context, "s", s);
		 //m_shader.set_vec3(context, "p", m_model_instance.light_position);
		//context->opengl.glDrawArrays(fan::opengl::GL_TRIANGLES, 0, 2916);
		sb_draw();
	}

	fan::graphics::animation::frame_transform_t get_keyframe() const {
		fan::graphics::animation::frame_transform_t kf;
		kf.position = get_position();
		kf.size = get_size();
		kf.angle = get_angle();
		return kf;
	}
	void set_keyframe(
		const fan::graphics::animation::frame_transform_t& ft
	) {
		//set_position(ft.position);
		set_size(ft.size);
		set_angle(ft.angle);
	}

protected:

	textures_t parse_model_texture(fastObjMesh* mesh) {

		textures_t textures;

		auto path_maker = [this](fan::string p) -> fan::string {

			auto found = p.find_last_of('\\');
			p = p.erase(0, found + 1);
			found = p.find('.');
			p.replace(found, p.size() - found, ".webp");

			return p;
		};

#if fan_debug >= fan_debug_low
		/*if (mesh->material_count > 1) {
			fan::throw_error("multi texture models not supported");
		}*/
#endif

		if (mesh->materials->map_Kd.path == nullptr) {
			//fan::print("no diffuse map");
			//return textures;
			//fan::throw_error("no diffuse map");
			textures.diffuse.load(gloco->get_context(), "models/color_map.webp");
		}
		else {
			fan::string diffuse_path = mesh->materials->map_Kd.path;
			auto loco = get_loco();
			textures.diffuse.load(loco->get_context(), "models/" + path_maker(diffuse_path));
		}
		/* if (mesh->materials->map_bump.path == nullptr) {
			 textures.depth.load(context, "models/missing_depth.webp");
		 }
		 else {
			 fan::string depth_path = mesh->materials->map_bump.path;
			 textures.depth.load(context, "models/" + path_maker(depth_path));
		 }*/
		return textures;
	}

	static void parse_model(loaded_model_t* lm, fastObjMesh* m) {

		for (unsigned int ii = 0; ii < m->group_count; ii++)
		{
			const fastObjGroup& grp = m->groups[ii];

			// reference https://github.com/thisistherk/fast_obj/blob/master/test/test.cpp

			uint32_t idx = 0;
			for (uint32_t i = 0; i < grp.face_count; i++) {

				uint32_t fv = m->face_vertices[grp.face_offset + i];

				for (uint32_t j = 0; j < fv; j++) {
					fastObjIndex mi = m->indices[grp.index_offset + idx];

					lm->vertices.push_back(fan::vec3(
						m->positions[3 * mi.p + 0],
						m->positions[3 * mi.p + 1],
						m->positions[3 * mi.p + 2]
					));

					lm->texture_coordinates.push_back(fan::vec2(
						m->texcoords[2 * mi.t + 0],
						m->texcoords[2 * mi.t + 1]
					));

					idx++;
				}
			}
		}
	}

	model_instance_t m_model_instance;

	fan::vec3* camera_position;

public:
	f32_t s = 2;
};

#include _FAN_PATH(graphics/opengl/2D/objects/hardcode_close.h)