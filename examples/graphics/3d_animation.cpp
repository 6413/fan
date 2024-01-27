#include fan_pch

namespace fan_3d {
  namespace animation {

    struct vertex_t {
      fan::vec3 position;
      fan::vec3 normal;
      fan::vec2 uv;
      fan::vec4 bone_ids;
      fan::vec4 bone_weights;
    };

    // structure to hold bone tree (skeleton)
    struct joint_t {
      int id; // position of the bone in final upload array
      std::string name;
      fan::mat4 offset;
      std::vector<joint_t> children;
    };

    // sturction representing an animation track
    struct bone_transform_track_t {
      std::vector<f32_t> position_timestamps;
      std::vector<f32_t> rotation_timestamps;
      std::vector<f32_t> scale_timestamps;

      std::vector<fan::vec3> positions;
      std::vector<fan::quat> rotations;
      std::vector<fan::vec3> scales;

      f32_t weight = 1.f;
    };

    // structure containing animation information
    struct animation_data_t {
      f_t duration;
      // ticks per second
      f_t tps;
      std::unordered_map<std::string, bone_transform_track_t> bone_transforms;
    };

    // a recursive function to read all bones and form skeleton
    bool read_skeleton(fan_3d::animation::joint_t& joint_t, aiNode* node, std::unordered_map<std::string, std::pair<int, fan::mat4>>& bone_info_table) {

      if (bone_info_table.find(node->mName.C_Str()) != bone_info_table.end()) { // if node is actually a bone
        joint_t.name = node->mName.C_Str();
        joint_t.id = bone_info_table[joint_t.name].first;
        joint_t.offset = bone_info_table[joint_t.name].second;

        for (int i = 0; i < node->mNumChildren; i++) {
          fan_3d::animation::joint_t child;
          read_skeleton(child, node->mChildren[i], bone_info_table);
          joint_t.children.push_back(child);
        }
        return true;
      }
      else { // find bones in children
        for (int i = 0; i < node->mNumChildren; i++) {
          if (read_skeleton(joint_t, node->mChildren[i], bone_info_table)) {
            return true;
          }

        }
      }
      return false;
    }

    void ProcessMaterial(aiMaterial* material, aiMesh* mesh) {
      // Access material properties as needed...

      // Access texture coordinates for the diffuse texture
      if (mesh->HasTextureCoords(0)) {
        // '0' represents the texture coordinate set, which may vary depending on the model.
        aiVector3D* textureCoords = mesh->mTextureCoords[0];

        // Iterate over vertices and print texture coordinates
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
          std::cout << "Vertex " << i << " - Texture Coordinates: "
            << textureCoords[i].x << ", " << textureCoords[i].y << std::endl;
        }
      }

      /*
      for(unsigned int i = 0; i < scene->mNumMaterials; i++)
{
    aiMaterial* material = scene->mMaterials[i];

    if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
    {
        aiString path;
        if(material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
        {
            int textureIndex = atoi(path.C_Str());
            aiTexture* texture = scene->mTextures[textureIndex];

            // The texture data is stored in the 'pcData' member of the aiTexture structure
            // The 'mWidth' member represents the size of the texture data in bytes
            unsigned char* textureData = reinterpret_cast<unsigned char*>(texture->pcData);

            // Now you can use 'textureData' as needed
        }
    }
}
      */

    }

    // pm -- parsed model

    struct pm_texture_data_t {
      // you either have textures in path or in binary data
      fan::vec2ui diffuse_texture_size;
      std::vector<uint8_t> diffuse_texture_data;
    };

    struct pm_model_data_t {
      std::vector<fan_3d::animation::vertex_t> vertices;
      fan_3d::animation::joint_t skeleton;
      uint32_t bone_count = 0;
    };

    struct parsed_model_t {
      pm_model_data_t model_data;
      pm_texture_data_t texture_data;
    };

    // converts indices to triangles only - todo use indices (needs bcol update)
    void load_model(const fan::string& root_path, const aiScene* scene, aiMesh* mesh, parsed_model_t& parsed_model) {
      std::vector<fan_3d::animation::vertex_t> temp_vertices;

      //load position, normal, uv
      for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
        //process position 
        fan_3d::animation::vertex_t vertex;
        fan::vec3 vector;
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.position = vector;
        //process normal
        vector.x = mesh->mNormals[i].x;
        vector.y = mesh->mNormals[i].y;
        vector.z = mesh->mNormals[i].z;
        vertex.normal = vector;
        //process uv
        fan::vec2 vec;
        vec.x = mesh->mTextureCoords[0][i].x;
        vec.y = mesh->mTextureCoords[0][i].y;
        vertex.uv = vec;

        vertex.bone_ids = fan::vec4i(0);
        vertex.bone_weights = fan::vec4(0.0f);

        temp_vertices.push_back(vertex);
      }

      std::unordered_map<std::string, std::pair<int, fan::mat4>> bone_info;
      std::vector<uint32_t> bone_counts;
      bone_counts.resize(temp_vertices.size(), 0);
      parsed_model.model_data.bone_count = mesh->mNumBones;

      //loop through each bone
      for (uint32_t i = 0; i < parsed_model.model_data.bone_count; i++) {
        aiBone* bone = mesh->mBones[i];

        bone_info[bone->mName.C_Str()] = { i, bone->mOffsetMatrix };

        //loop through each vertex that have that bone
        for (int j = 0; j < bone->mNumWeights; j++) {
          uint32_t id = bone->mWeights[j].mVertexId;
          f32_t weight = bone->mWeights[j].mWeight;
          bone_counts[id]++;
          switch (bone_counts[id]) {
            case 1:
              temp_vertices[id].bone_ids.x = i;
              temp_vertices[id].bone_weights.x = weight;
              break;
            case 2:
              temp_vertices[id].bone_ids.y = i;
              temp_vertices[id].bone_weights.y = weight;
              break;
            case 3:
              temp_vertices[id].bone_ids.z = i;
              temp_vertices[id].bone_weights.z = weight;
              break;
            case 4:
              temp_vertices[id].bone_ids.w = i;
              temp_vertices[id].bone_weights.w = weight;
              break;
            default:
              // ignore above 4 - i dont know what they are
              //fan::throw_error("invalid bone_counts id");
              break;

          }
        }
      }

      //normalize weights to make all weights sum 1
      for (int i = 0; i < temp_vertices.size(); i++) {
        fan::vec4& bone_weights = temp_vertices[i].bone_weights;
        f32_t total_weight = bone_weights.plus();
        if (total_weight > 0.0f) {
          temp_vertices[i].bone_weights = fan::vec4(
            bone_weights.x / total_weight,
            bone_weights.y / total_weight,
            bone_weights.z / total_weight,
            bone_weights.w / total_weight
          );
        }
      }

      {
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
        {
          aiString path;
          if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
          {
            if (scene->mTextures == nullptr) {
              fan::webp::image_info_t ii;
              if (fan::webp::load(root_path + path.C_Str(), &ii)) {
                fan::throw_error("failed to load image data from path:" + root_path + path.C_Str());
              }

              static constexpr int channels_rgba = 4;
              parsed_model.texture_data.diffuse_texture_data.insert(
                parsed_model.texture_data.diffuse_texture_data.end(),
                (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * channels_rgba
              );
              parsed_model.texture_data.diffuse_texture_size = ii.size;
              //parsed_model.texture_data.diffuse_texture_path_list.push_back(root_path + path.C_Str());

              // use with non combined binary
            }
            else {
              int textureIndex = atoi(path.C_Str());
              aiTexture* texture = scene->mTextures[textureIndex];

              unsigned char* textureData = reinterpret_cast<unsigned char*>(texture->pcData);
              // compressed image format -- mWidth means size of bytes
              if (texture->mHeight == 0) {
                fan::throw_error("doesnt support compressed textures");
              }
              else {
                // todo needs byte calculate width * height * channel count - how to get it from assimp?
                //parsed_model.texture_data.insert(textureData, textureData + texture->)
              }
            }
          }
        }
      }

      //load indices
      for (int i = 0; i < mesh->mNumFaces; i++) {
        aiFace& face = mesh->mFaces[i];
        for (uint32_t j = 0; j < face.mNumIndices; j++) {
          parsed_model.model_data.vertices.push_back(temp_vertices[face.mIndices[j]]);
        }
      }

      // create bone hirerchy
      read_skeleton(parsed_model.model_data.skeleton, scene->mRootNode, bone_info);
    }
    void load_animation(const aiScene* scene, fan_3d::animation::animation_data_t& animation) {
      //loading  first Animation
      aiAnimation* anim = scene->mAnimations[0];

      animation.duration = anim->mDuration/* * anim->mTicksPerSecond*/;

      for (int i = 0; i < anim->mNumChannels; i++) {
        aiNodeAnim* channel = anim->mChannels[i];
        fan_3d::animation::bone_transform_track_t track;
        for (int j = 0; j < channel->mNumPositionKeys; j++) {
          track.position_timestamps.push_back(channel->mPositionKeys[j].mTime);
          track.positions.push_back(channel->mPositionKeys[j].mValue);
        }
        for (int j = 0; j < channel->mNumRotationKeys; j++) {
          track.rotation_timestamps.push_back(channel->mRotationKeys[j].mTime);
          track.rotations.push_back(channel->mRotationKeys[j].mValue);

        }
        for (int j = 0; j < channel->mNumScalingKeys; j++) {
          track.scale_timestamps.push_back(channel->mScalingKeys[j].mTime);
          track.scales.push_back(channel->mScalingKeys[j].mValue);

        }
        animation.bone_transforms[channel->mNodeName.C_Str()] = track;
      }
    }

    std::pair<uint32_t, f32_t> get_time_fraction(const std::vector<f32_t>& times, f32_t dt) {
      if (times.empty()) {
        return { -1, 0 };
      }
      auto it = std::upper_bound(times.begin(), times.end(), dt);
      uint32_t segment = std::distance(times.begin(), it);
      if (segment == 0) {
        segment++;
        ++it;
        if (times.size() <= segment) {
          fan::throw_error("not enough keyframe data");
        }
      }
      f32_t start = *(it - 1);
      f32_t end = *it;
      return { segment, (dt - start) / (end - start) };
    }

    void apply_transform(fan_3d::animation::joint_t& bone, fan::mat4& rot_transform) {
      static std::unordered_map<int, fan::vec3> v;
      //fan::vec3 ang = 0;
      ImGui::DragFloat3((bone.name + " angle").c_str(), v[bone.id].data(), 0.1);

      if (v[bone.id].y != 0) {
        rot_transform = rot_transform.rotate(v[bone.id].y, fan::vec3(0, std::abs(v[bone.id].y), 0));
      }
      if (v[bone.id].z != 0) {
        rot_transform = rot_transform.rotate(v[bone.id].z, fan::vec3(0, 0, std::abs(v[bone.id].z)));
      }
      if (v[bone.id].x != 0) {
        rot_transform = rot_transform.rotate(v[bone.id].x, fan::vec3(std::abs(v[bone.id].x), 0, 0));
      }
    }

    // fan model stuff
    struct fms_t {
      fms_t(const std::string& path)
      {
        Assimp::Importer importer;

        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
          fan::print("animation load error:", importer.GetErrorString());
          exit(1);
        }

        aiMesh* mesh = scene->mMeshes[0];

        m_transform = scene->mRootNode->mTransformation;

        std::string root_path = "";

        size_t last_slash_pos = path.find_last_of("/");
        if (last_slash_pos != std::string::npos) {
          root_path = path.substr(0, last_slash_pos + 1);
        }

        load_model(root_path, scene, mesh, parsed_model);
        load_animation(scene, m_animation);

        m_current_pose.resize(parsed_model.model_data.bone_count, fan::mat4(1)); // use this for no animation

        m_modified_verticies = parsed_model.model_data.vertices;
      }

      void calculate_modified_vertices() {
        for (int i = 0; i < parsed_model.model_data.vertices.size(); ++i) {

          auto bone_transforms = m_current_pose;
          fan::vec4i bone_ids = parsed_model.model_data.vertices[i].bone_ids;
          fan::vec4 bone_weights = parsed_model.model_data.vertices[i].bone_weights;

          fan::mat4 boneTransform = fan::mat4(0.0);
          boneTransform += bone_transforms[bone_ids.x] * bone_weights.x;
          boneTransform += bone_transforms[bone_ids.y] * bone_weights.y;
          boneTransform += bone_transforms[bone_ids.z] * bone_weights.z;
          boneTransform += bone_transforms[bone_ids.w] * bone_weights.w;
          fan::vec4 pos = boneTransform * fan::vec4(parsed_model.model_data.vertices[i].position, 1.0);

          fan::mat4 model_matrix = fan::mat4(1);

          fan::vec4 result = pos;
          m_modified_verticies[i].position = fan::vec3(result.x, result.y, result.z);
        }
      }

      // parses joint data @ dt
      void fk_parse_joint_data(const bone_transform_track_t& btt, fan::vec3& position, fan::quat& rotation, fan::vec3& scale, f32_t dt) {
        std::pair<uint32_t, f32_t> fp;
        {// position
          fp = get_time_fraction(btt.position_timestamps, dt);
          if (fp.first != (uint32_t)-1) {
            fan::vec3 p1 = btt.positions[fp.first - 1];
            fan::vec3 p2 = btt.positions[fp.first];
            fan::vec3 pos = fan::mix(p1, p2, fp.second);
            position += pos * btt.weight;
          }
        }

        {// rotation
          fp = get_time_fraction(btt.rotation_timestamps, dt);
          if (fp.first != (uint32_t)-1) {
            fan::quat rotation1 = btt.rotations[fp.first - 1];
            fan::quat rotation2 = btt.rotations[fp.first];
            fan::quat rot = fan::quat::slerp(rotation1, rotation2, fp.second);

            fan::quat slerped_quat = fan::quat::slerp(fan::quat(1, 0, 0, 0), rot, btt.weight);
            rotation = (rotation * slerped_quat).normalize();
          }
        }

        { // size
          fp = get_time_fraction(btt.scale_timestamps, dt);
          if (fp.first != (uint32_t)-1) {
            fan::vec3 s1 = btt.scales[fp.first - 1];
            fan::vec3 s2 = btt.scales[fp.first];
            scale += fan::mix(s1, s2, fp.second) * btt.weight;
          }
        }
      }

      void get_pose(f32_t dt, joint_t& joint, fan::mat4& parent_transform) {

        dt = fmod(dt, m_animation.duration);
        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;

        auto found = m_animation.bone_transforms.find(joint.name);
        f32_t prev_weight = found->second.weight;
        if (found == m_animation.bone_transforms.end()) {
          fan::throw_error("invalid bone data");
        }
        else {
          found->second.weight = 1.f;
          fk_parse_joint_data(found->second, position, rotation, scale, dt);
          found->second.weight = prev_weight;
        }

        fan::mat4 mtranslation = fan::mat4(1).translate(position);
        fan::mat4 mrotation = fan::mat4(rotation);
        mrotation = mrotation.rotate(fan::math::pi / 2, fan::vec3(1, 0, 0));
        fan::mat4 mscale = fan::mat4(1).scale(scale);

        fan::mat4 local_transform = mtranslation * mrotation * mscale;
        fan::mat4 global_transform = parent_transform * local_transform;

        m_current_pose[joint.id] = m_transform * global_transform * joint.offset;

        for (fan_3d::animation::joint_t& child : joint.children) {
          get_pose(dt, child, global_transform);
        }
      }
      void get_pose(f32_t dt) {
        fan::mat4 intial(1);
        get_pose(dt, parsed_model.model_data.skeleton, intial);
      }

      void fk_get_pose(f32_t dt, joint_t& joint, fan::mat4& parent_transform) {

        // if there are no additional animations, use default animation 
        // alternatively use with dt0 to have "T-pose"
        if (animation_list.empty()) {
          get_pose(dt, joint, parent_transform);
          return;
        }

        dt = fmod(dt, m_animation.duration);
        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;

        {
          auto found = m_animation.bone_transforms.find(joint.name);
          if (found == m_animation.bone_transforms.end()) {
            fan::throw_error("invalid bone data");
          }
          else {
            fk_parse_joint_data(found->second, position, rotation, scale, dt);
          }
        }

        for (auto& anim : animation_list) {
          auto& anim_data = anim.second;
          auto found = anim_data.bone_transforms.find(joint.name);
          if (found == m_animation.bone_transforms.end()) {
            fan::throw_error("invalid bone data");
          }
          else {
            fk_parse_joint_data(found->second, position, rotation, scale, dt);
          }
        }

        fan::mat4 mtranslation = fan::mat4(1).translate(position);
        fan::mat4 mrotation = fan::mat4(rotation);
        fan::mat4 mscale = fan::mat4(1).scale(scale);

        fan::mat4 local_transform = mtranslation * mrotation * mscale;
        fan::mat4 global_transform = parent_transform * local_transform;

        m_current_pose[joint.id] = m_transform * global_transform * joint.offset;

        for (fan_3d::animation::joint_t& child : joint.children) {
          fk_get_pose(dt, child, global_transform);
        }
      }
      void fk_get_pose(f32_t dt) {
        fan::mat4 intial(1);
        fk_get_pose(dt, parsed_model.model_data.skeleton, intial);
      }

      struct one_triangle_t {
        fan::vec3 p[3];
        fan::vec2 tc[3];
      };

      void get_triangle_vec(uint32_t mesh_id, std::vector<one_triangle_t>* triangles) {
        static constexpr int edge_count = 3;
        // ignore i for now since only one scene
        triangles->resize(m_modified_verticies.size() / edge_count);
        for (int i = 0; i < m_modified_verticies.size(); ++i) {
          //                                                      is it illegal | modified vertices or m_vertex
          (*triangles)[i / edge_count].p[i % edge_count] = *(fan::vec3*)&m_modified_verticies[i].position;
          (*triangles)[i / edge_count].tc[i % edge_count] = *(fan::vec3*)&m_modified_verticies[i].uv;
        }
      }

      void iterate_joints(joint_t& joint, auto lambda) {
        lambda(joint);
        for (fan_3d::animation::joint_t& child : joint.children) {
          iterate_joints(child, lambda);
        }
      }

      void get_bone_names(auto lambda) {
        iterate_joints(parsed_model.model_data.skeleton, [&](joint_t& joint) {lambda(joint.name); });
      }

      static constexpr int invalid_bone = -1;
      int get_bone_id_by_name(const fan::string& name) {
        bool found = false;
        int bone_id = invalid_bone;
        iterate_joints(parsed_model.model_data.skeleton,
          [&](joint_t& joint) {
            if (found) {
              return;
            }
            if (name == joint.name) {
              found = true;
              bone_id = joint.id;
            }
          });
        return bone_id;
      }

      using anim_key_t = std::string;

      std::unordered_map<anim_key_t, animation_data_t> animation_list;

      fan::string create_an(const fan::string& key, f32_t weight) {
        auto& node = animation_list[key];
        //node = m_animation;
        node.duration = m_animation.duration;

        iterate_joints(parsed_model.model_data.skeleton,
          [&](joint_t& joint) {
            node.bone_transforms[joint.name];
            m_animation.bone_transforms[joint.name].weight = 1.f / (animation_list.size() + 1.f);
            for (auto& i : animation_list) {
              i.second.bone_transforms[joint.name].weight = 1.f / (animation_list.size() + 1.f);
            }
          });
        return key;
      }

      uint32_t fk_set_rot(const anim_key_t& key, f32_t dt, const fan::string& bone_id, const fan::vec3& angle) {
        auto& node = animation_list[key];
        auto found = node.bone_transforms.find(bone_id);
        if (found == node.bone_transforms.end()) {
          fan::throw_error("could not find bone:" + bone_id);
        }
        auto& transform = found->second;
        //auto it = std::upper_bound(transform.rotation_timestamps.begin(), transform.rotation_timestamps.end(), dt * 1000.f); // convert to ms
        //uint32_t index = std::distance(transform.rotation_timestamps.begin(), it);

        //transform.rotations[index - 1] = /*m_animation.bone_transforms[bone_id].rotations[index - 1] + */fan::quat::from_euler(angle);
        //transform.weight = 0.5;
        // todo fix
        //m_animation.bone_transforms[bone_id].weight = 0.5;

        transform.position_timestamps.push_back(dt * 1000.f);
        transform.rotation_timestamps.push_back(dt * 1000.f);
        transform.scale_timestamps.push_back(dt * 1000.f);
        transform.positions.push_back(0);
        transform.rotations.push_back(fan::quat::from_euler(angle));
        transform.scales.push_back(1);

        return 0;
      }

      parsed_model_t parsed_model;
      fan_3d::animation::animation_data_t m_animation;

      // custom poses
      std::vector<fan_3d::animation::vertex_t> m_modified_verticies;

      std::vector<fan::mat4> m_current_pose;

      fan::mat4 m_transform;
    };

    struct animation_t {

      std::string animation_vs = R"(
					#version 440 core
					layout (location = 0) in vec3 vertex; 
					layout (location = 1) in vec3 normal;
					layout (location = 2) in vec2 uv;
					layout (location = 3) in vec4 bone_ids;
					layout (location = 4) in vec4 bone_weights;

					out vec2 tex_coord;
					out vec3 v_normal;
					out vec3 v_pos;
					out vec4 bw;

					uniform mat4 projection;
					uniform mat4 view;

					void main()
					{
            mat4 model = mat4(1);
						gl_Position = projection * view * model * vec4(vertex, 1.0);
            tex_coord = uv;
						v_normal = mat3(transpose(inverse(model))) * normal;
						v_normal = normalize(v_normal);
					}
			)";

      std::string animation_fs = R"(
					#version 440 core

					in vec2 tex_coord;
					in vec3 v_normal;
					in vec3 v_pos;
					in vec4 bw;
					out vec4 color;

					uniform sampler2D diff_texture;
	
					void main()
					{
						color = vec4(texture(diff_texture, tex_coord).rgb, 1);
            //color = vec4(gl_FragCoord.xy / 4098, 0, 1);
					}
			)";

      animation_t(const fan::string& path) : fms(path) {
        fan::webp::image_info_t ii;
        ii.data = fms.parsed_model.texture_data.diffuse_texture_data.data();
        ii.size = fms.parsed_model.texture_data.diffuse_texture_size;
        image.load(ii);
        //image.load(fms.parsed_model.texture_data.diffuse_texture_path_list[0]);
        //fms.parsed_model.texture_data.
        m_shader.open();
        m_shader.set_vertex(
          animation_vs
        );

        m_shader.set_fragment(
          animation_fs
        );

        m_shader.compile();

        auto& context = gloco->get_context();
        vao.open(context);
        vbo.open(context, fan::opengl::GL_ARRAY_BUFFER);
        vao.bind(context);

        upload_modified_vertices();

        gloco->get_context().opengl.glEnableVertexAttribArray(0);
        gloco->get_context().opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::animation::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::animation::vertex_t, position));
        gloco->get_context().opengl.glEnableVertexAttribArray(1);
        gloco->get_context().opengl.glVertexAttribPointer(1, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::animation::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::animation::vertex_t, normal));
        gloco->get_context().opengl.glEnableVertexAttribArray(2);
        gloco->get_context().opengl.glVertexAttribPointer(2, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::animation::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::animation::vertex_t, uv));
        gloco->get_context().opengl.glEnableVertexAttribArray(3);
        gloco->get_context().opengl.glVertexAttribPointer(3, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::animation::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::animation::vertex_t, bone_ids));
        gloco->get_context().opengl.glEnableVertexAttribArray(4);
        gloco->get_context().opengl.glVertexAttribPointer(4, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::animation::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::animation::vertex_t, bone_weights));
        gloco->get_context().opengl.glBindVertexArray(0);
      }

      void upload_modified_vertices() {
        vao.bind(gloco->get_context());
        vbo.write_buffer(
          gloco->get_context(),
          &fms.m_modified_verticies[0],
          sizeof(fan_3d::animation::vertex_t) * fms.m_modified_verticies.size()
        );
      }

      void draw() {
        m_shader.use();

        fan::mat4 projection(1);
        static constexpr f32_t fov = 90.f;
        projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)gloco->window.get_size().x / (f32_t)gloco->window.get_size().y, 0.1f, 1000.0f);
        fan::mat4 view(m_camera.get_view_matrix());

        m_shader.set_mat4("projection", projection);
        m_shader.set_mat4("view", view);


        gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
        image.bind_texture();

        m_shader.set_int("diff_texture", 0);

        auto& context = gloco->get_context();
        context.opengl.glDisable(fan::opengl::GL_BLEND);
        vao.bind(context);

        gloco->get_context().opengl.glDrawArrays(fan::opengl::GL_TRIANGLES, 0, fms.m_modified_verticies.size());
      }

      fms_t fms;

      fan::camera m_camera;
      loco_t::shader_t m_shader;
      loco_t::image_t image;
      fan::opengl::core::vao_t vao;
      fan::opengl::core::vbo_t vbo;
    };
  }
}

int main() {
  //fan::window_t::set_flag_value<fan::window_t::flags::no_mouse>(true);
  loco_t loco;
  //loco.window.lock_cursor_and_set_invisible(true);


  loco.set_vsync(0);

  struct triangle_list_t {
    uint32_t matid;
    std::vector<fan_3d::animation::fms_t::one_triangle_t> triangle_vec;
  };

  fan_3d::animation::animation_t animation("models/model2.dae");

  // calculate first pose (initialize modified vertices)

  //fms.get_pose(1);
  //fms.calculate_modified_vertices();

  std::vector<triangle_list_t> triangles;

  static constexpr int nscenes = 1;
  for (uintptr_t i = 0; i < nscenes; i++) {
    triangle_list_t tl;
    //tl.matid = fms.get_material_id(i);
    animation.fms.get_triangle_vec(i, &tl.triangle_vec);
  }

  animation.fms.get_bone_names([](const std::string& name) {
    fan::print(name);
    });

  // auto boneid = animation.fms.get_bone_id_by_name("Armature_Upper_Arm_R");

  auto anid = animation.fms.create_an("an_name", 0.5);
  auto anid2 = animation.fms.create_an("an_name2", 0.5);

  auto animation_node_id = animation.fms.fk_set_rot(anid, 0.3/* time in seconds */, "Armature_Lower_Arm_R", fan::vec3(0, 180, 0));
  auto animation_node_id2 = animation.fms.fk_set_rot(anid, 0.4/* time in seconds */, "Armature_Lower_Arm_R", fan::vec3(0, 180, 0));


  auto animation_node_id3 = animation.fms.fk_set_rot(anid2, 0.5/* time in seconds */, "Armature_Lower_Leg_L", fan::vec3(0, 180, 0));
  auto animation_node_id4 = animation.fms.fk_set_rot(anid2, 0.7/* time in seconds */, "Armature_Lower_Leg_L", fan::vec3(0, 180, 0));

  //auto animation_node_id2 = fms.fk_set_rot(anid, 0.3 *2/* time in seconds */, "Upper_Leg_L", fan::vec3(90, 180, 0));

  //fms.an.setweight(anid, animation_node_id, 1);
  //// or
  //fms.an.setweight("an_name", animation_node_id, 1);

  //fms.an.setprevinter(anid, animation_node_id, fms_t::INTER_SINE_WAVE);
  //fms.an.setnextinter(anid, animation_node_id, fms_t::INTER_SINE_WAVE);

  animation.m_camera.position = { 3.46, 1.94, -6.22 };
  //fan_3d::graphics::add_camera_rotation_callback(&camera);
  loco.window.add_key_callback(fan::key_escape, fan::keyboard_state::press, [](const auto&) { exit(0); });

  fan::time::clock timer;
  timer.start();
  gloco->m_draw_queue_light.push_back([&] {
    // convert ns to s then to ms
    static f32_t divider = 2;
    animation.fms.get_pose(timer.elapsed() / 1e+6 / divider + 0.0001);
    animation.fms.calculate_modified_vertices();
    animation.upload_modified_vertices();
    animation.draw();
    });

  loco.window.add_mouse_motion([&](const auto& d) {
    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
      animation.m_camera.rotate_camera(d.motion);
    }
    });

  loco.loop([&] {
    animation.m_camera.move(100);
    fan::print(animation.m_camera.position);

    if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
      animation.m_camera.rotate_camera(fan::vec2(-0.01, 0));
    }
    if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) {
      animation.m_camera.rotate_camera(fan::vec2(0.01, 0));
    }
    if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
      animation.m_camera.rotate_camera(fan::vec2(0, -0.01));
    }
    if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
      animation.m_camera.rotate_camera(fan::vec2(0, 0.01));
    }

    loco.get_fps();
    });
}