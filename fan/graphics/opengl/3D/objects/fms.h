#pragma once

namespace fan_3d {
  namespace model {
    using namespace fan::opengl;

    struct vertex_t {
      fan::vec3 position;
      fan::vec3 normal;
      fan::vec2 uv;
      fan::vec4i bone_ids;
      fan::vec4 bone_weights;
      fan::vec3 tangent;
      fan::vec3 bitangent;
    };

    struct bone_transform_track_t {
      std::vector<f32_t> position_timestamps;
      std::vector<f32_t> rotation_timestamps;
      std::vector<f32_t> scale_timestamps;

      std::vector<fan::vec3> positions;
      std::vector<fan::quat> rotations;
      std::vector<fan::vec3> scales;

      f32_t weight = 1.f;
    };

    struct animation_data_t {
      struct bone_pose_t {
        fan::vec3 position = 0;
        fan::quat rotation;
        fan::vec3 scale = 1;
      };

      f_t duration = 1.f;
      f32_t weight = 0;
      struct type_e {
        enum {
          invalid = -1,
          nonlinear_animation,
          custom
        };
      };
      uint32_t type = type_e::invalid;
      std::vector<bone_pose_t> bone_poses;
      std::unordered_map<std::string, bone_transform_track_t> bone_transforms;
    };

    struct bone_t {
      int id = -1;
      std::string name;
      fan::mat4 offset;
      fan::mat4 transform;
      fan::mat4 global_transform;
      bone_t* parent;
      std::vector<bone_t*> children;

      fan::vec3 translation = 0;
      fan::vec3 rotation = 0;
      fan::vec3 scale = 1;
    };

    struct mesh_t {
      std::vector<fan_3d::model::vertex_t> vertices;
      std::vector<uint32_t> indices;

      fan::opengl::core::vao_t VAO;
      fan::opengl::core::vbo_t VBO;
      fan::opengl::GLuint EBO;
      std::string texture_names[AI_TEXTURE_TYPE_MAX + 1]{};
    };

    // pm -- parsed model
    struct pm_texture_data_t {
      fan::vec2ui size = 0;
      std::vector<uint8_t> data;
      int channels = 0;
    };
    inline static std::unordered_map<std::string, pm_texture_data_t> cached_texture_data;

    struct pm_material_data_t {
      fan::vec4 color[AI_TEXTURE_TYPE_MAX + 1];
    };

    struct fms_model_info_t {
      fan::string path;
      std::string texture_path;
      int use_cpu = false;
    };

    // fan model stuff
    struct fms_t {
      fms_t() = default;
      fms_t(const fms_model_info_t& fmi) {
        if (!load_model(fmi.path)) {
          fan::throw_error("failed to load model:" + fmi.path);
        }
        calculated_meshes = meshes;
        p = fmi;
      }
      
      // ---------------------model hierarchy---------------------

      mesh_t process_mesh(aiMesh* mesh) {
        mesh_t new_mesh;
        std::vector<vertex_t> temp_vertices(mesh->mNumVertices);

        for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
          vertex_t& vertex = temp_vertices[i];
          vertex.position = fan::vec3(
            mesh->mVertices[i].x,
            mesh->mVertices[i].y,
            mesh->mVertices[i].z
          );
          if (mesh->HasNormals()) {
            vertex.normal = fan::vec3(
              mesh->mNormals[i].x,
              mesh->mNormals[i].y,
              mesh->mNormals[i].z
            );
          }
          if (mesh->mTextureCoords[0]) {
            vertex.uv = fan::vec2(
              mesh->mTextureCoords[0][i].x,
              mesh->mTextureCoords[0][i].y
            );
          }
          else {
            vertex.uv = fan::vec2(0.0f);
          }

          vertex.bone_ids = fan::vec4i(-1);
          vertex.bone_weights = fan::vec4(0.0f);
        }
        // process bones using a running maximum approach
        for (uint32_t i = 0; i < mesh->mNumBones; i++) {
          aiBone* bone = mesh->mBones[i];
          std::string bone_name = bone->mName.C_Str();

          auto it = bone_map.find(bone_name);
          if (it == bone_map.end()) {
            continue;
          }
          int boneId = it->second->id;

          for (uint32_t j = 0; j < bone->mNumWeights; j++) {
            uint32_t vertexId = bone->mWeights[j].mVertexId;
            f32_t weight = bone->mWeights[j].mWeight;

            // find the slot with minimum weight and replace if current weight is larger
            int min_index = 0;
            f32_t min_weight = temp_vertices[vertexId].bone_weights[0];

            for (int k = 1; k < 4; k++) {
              if (temp_vertices[vertexId].bone_weights[k] < min_weight) {
                min_weight = temp_vertices[vertexId].bone_weights[k];
                min_index = k;
              }
            }

            if (weight > min_weight) {
              temp_vertices[vertexId].bone_weights[min_index] = weight;
              temp_vertices[vertexId].bone_ids[min_index] = boneId;
            }
          }
        }

        // normalize weights
        for (auto& vertex : temp_vertices) {
          f32_t sum = vertex.bone_weights.x + vertex.bone_weights.y +
            vertex.bone_weights.z + vertex.bone_weights.w;

          if (sum > 0.0f) {
            vertex.bone_weights /= sum;
          }
          else {
            // if no bones influence this vertex, assign it to the first bone
            vertex.bone_ids.x = 0;
            vertex.bone_weights.x = 1.0f;
          }
        }

        new_mesh.vertices = temp_vertices;

        for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
          aiFace& face = mesh->mFaces[i];
          for (uint32_t j = 0; j < face.mNumIndices; j++) {
            new_mesh.indices.push_back(face.mIndices[j]);
          }
        }
        return new_mesh;
      }

      std::string load_textures(mesh_t& mesh, aiMesh* ai_mesh) {
        if (scene->mNumMaterials == 0) {
          return "";
        }
        static constexpr auto textures_to_load = std::to_array({
          aiTextureType_DIFFUSE,
          aiTextureType_AMBIENT,
          aiTextureType_SPECULAR,
          aiTextureType_EMISSIVE,
          aiTextureType_SHININESS,
          aiTextureType_METALNESS
        });
        aiMaterial* material = scene->mMaterials[ai_mesh->mMaterialIndex];
        for (const aiTextureType& texture_type : textures_to_load) {
          aiString path;
          if (material->GetTexture(texture_type, 0, &path) != AI_SUCCESS) {
            return "";
          }

          auto embedded_texture = scene->GetEmbeddedTexture(path.C_Str());

          if (embedded_texture && embedded_texture->mHeight == 0) {
            int width, height, nr_channels;
            unsigned char* data = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(embedded_texture->pcData), embedded_texture->mWidth, &width, &height, &nr_channels, 0);
            if(data == nullptr){
              fan::throw_error("failed to load texture");
            }

            // must not collide with other names
            std::string generated_str = path.C_Str() + std::to_string(texture_type);
            mesh.texture_names[texture_type] = generated_str;
            auto& td = fan_3d::model::cached_texture_data[generated_str];
            td.size = fan::vec2(width, height);
            td.data.insert(td.data.end(), data, data + td.size.multiply() * nr_channels);
            td.channels = nr_channels;
            stbi_image_free(data);
            return std::string(path.C_Str());
          }
          else {
            fan::string file_path = p.texture_path + "/" + scene->GetShortFilename(path.C_Str());
            mesh.texture_names[texture_type] = file_path;
            auto found = cached_texture_data.find(file_path);
            if (found == cached_texture_data.end()) {
              fan::image::image_info_t ii;
              fan::image::load(file_path, &ii);
              auto& td = cached_texture_data[file_path];
              td.size = ii.size;
              td.data.insert(td.data.end(), (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * ii.channels);
              td.channels = ii.channels;
              fan::image::free(&ii);
              return file_path;
            }
            return found->first;
          }
        }
        return "";
      }
      pm_material_data_t load_materials(aiMesh* ai_mesh) {
        pm_material_data_t material_data;
        for(uint32_t i = 0; i <= AI_TEXTURE_TYPE_MAX; i++){
          material_data.color[i] = fan::vec4(1, 0, 1, 1);
        }

        if (scene->mNumMaterials <= ai_mesh->mMaterialIndex) {
          return material_data;
        }

        aiMaterial* cmaterial = scene->mMaterials[ai_mesh->mMaterialIndex];
        cmaterial->Get(AI_MATKEY_COLOR_DIFFUSE, material_data.color[aiTextureType_DIFFUSE]);
        cmaterial->Get(AI_MATKEY_COLOR_AMBIENT, material_data.color[aiTextureType_AMBIENT]);
        cmaterial->Get(AI_MATKEY_COLOR_SPECULAR, material_data.color[aiTextureType_SPECULAR]);
        cmaterial->Get(AI_MATKEY_COLOR_EMISSIVE, material_data.color[aiTextureType_EMISSIVE]);

        return material_data;
      }

      void process_skeleton(aiNode* node, bone_t* parent) {
        if (parent == nullptr) {
          bone_count = 0;
        }
        bone_t* bone = new bone_t();
        bone->name = node->mName.C_Str();
        bone->id = bone_count;
        bone->parent = parent;
        bone->transform = node->mTransformation;
        bone->offset = fan::mat4(1.0f);
        bone_map[bone->name] = bone;
        ++bone_count;
        bone_strings.push_back(bone->name);
        if (parent == nullptr) {
          root_bone = bone;
        }
        else {
          parent->children.push_back(bone);
        }
        for (uint32_t i = 0; i < node->mNumChildren; i++) {
          process_skeleton(node->mChildren[i], bone);
        }
      }

      void process_bone_offsets(aiMesh* mesh) {
        for (uint32_t i = 0; i < mesh->mNumBones; i++) {
          aiBone* bone = mesh->mBones[i];
          std::string bone_name = bone->mName.C_Str();

          auto it = bone_map.find(bone_name);
          if (it != bone_map.end()) {
            it->second->offset = bone->mOffsetMatrix;
          }
        }
      }
      void update_bone_rotation(const std::string& boneName, const fan::vec3& rotation) {
        auto it = bone_map.find(boneName);
        if (it != bone_map.end()) {
          bone_t* bone = it->second;
          bone->rotation = rotation;
        }
      }
      void update_bone_transforms_impl(bone_t* bone, const fan::mat4& parentTransform) {
        if (!bone) return;


        fan::mat4 translation = fan::mat4(1).translate(bone->translation);
        fan::mat4 rotation = fan::mat4(1).rotate(bone->rotation);
        fan::mat4 scale = fan::mat4(1).scale(bone->scale);

        fan::mat4 node_transform = bone->transform;

        fan::mat4 globalTransform = 
          parentTransform * 
          node_transform * 
          (translation * rotation * scale)
        ;

        bone_transforms[bone->id] = globalTransform  * bone->offset;

        for (bone_t* child : bone->children) {
          update_bone_transforms_impl(child, globalTransform);
        }
      }
      void update_bone_transforms() {
        bone_transforms.clear();
        bone_transforms.resize(bone_count);
        if (root_bone) {
          update_bone_transforms_impl(root_bone, m_transform);
        }
      }
      void iterate_bones(bone_t& bone, auto lambda) {
        lambda(bone);
        for (bone_t* child : bone.children) {
          iterate_bones(*child, lambda);
        }
      }
      fan::vec4 calculate_bone_transform(const std::vector<fan::mat4>& bt, uint32_t mesh_id, uint32_t vertex_id) {
        auto& vertex = meshes[mesh_id].vertices[vertex_id];

        fan::vec4 totalPosition(0.0);

        for (int i = 0; i < 4; i++) {
          f32_t weight = vertex.bone_weights[i];
          int boneId = vertex.bone_ids[i];
          if (boneId == -1) {
            continue;
          }

          if (boneId >= bt.size()) {
            totalPosition = fan::vec4(vertex.position, 1.0);
            break;
          }
          fan::vec4 local_position = bt[boneId] * fan::vec4(vertex.position, 1.0);
          totalPosition += local_position * weight;
        }

        fan::mat4 model(1);
        fan::vec4 worldPos = model * totalPosition;
        fan::vec4 gl_Position = worldPos;
        return gl_Position;
      }

      bool load_model(const std::string& path) {
        Assimp::Importer importer;

        scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
          return false;
        }

        m_transform = scene->mRootNode->mTransformation;

        process_skeleton(scene->mRootNode, nullptr);
        load_animations();

        meshes.clear();

        for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
          mesh_t mesh = process_mesh(scene->mMeshes[i]);
          aiMesh* ai_mesh = scene->mMeshes[i];
          pm_material_data_t material_data = load_materials(ai_mesh);
          load_textures(mesh, ai_mesh);
          material_data_vector.push_back(material_data);
          process_bone_offsets(ai_mesh);
          meshes.push_back(mesh);
        }

        update_bone_transforms();
        return true;
      }

      void calculate_vertices(const std::vector<fan::mat4>& bt, uint32_t mesh_id, const fan::mat4& model) {
        for (int i = 0; i < meshes[mesh_id].vertices.size(); ++i) {
          fan::vec3 v = meshes[mesh_id].vertices[i].position;
          if (bt.empty()) {
            fan::vec4 vertex_position = m_transform * model * fan::vec4(v, 1.0);
            calculated_meshes[mesh_id].vertices[i].position = fan::vec3(vertex_position.x, vertex_position.y, vertex_position.z);
          }
          else {
            fan::vec4 interpolated_bone_transform = calculate_bone_transform(bt, mesh_id, i);
            fan::vec4 vertex_position = fan::vec4(v, 1.0);
            fan::vec4 result = model * interpolated_bone_transform;

            calculated_meshes[mesh_id].vertices[i].position = fan::vec3(result.x, result.y, result.z);
          }
        }
      }

      // flag default, ik, fk?
      void calculate_modified_vertices(const fan::mat4& model = fan::mat4(1)) {
        for (uint32_t i = 0; i < meshes.size(); ++i) {
          update_bone_transforms();
          calculate_vertices(bone_transforms, i, model);
        }
      }

      // triangle consists of 3 vertices
      struct one_triangle_t {
        fan::vec3 position[3]{};
        fan::vec3 normal[3]{};
        fan::vec2 uv[3]{};
        uint32_t vertex_indices[3];
      };

    std::vector<one_triangle_t> get_triangles(uint32_t mesh_id) {
      std::vector<one_triangle_t> triangles;
      static constexpr int edge_count = 3;
      const auto& mesh = calculated_meshes[mesh_id];

      size_t triangle_count = mesh.indices.size() / edge_count;
      triangles.resize(triangle_count);

      for (size_t i = 0; i < triangle_count; ++i) {
        for (int j = 0; j < edge_count; ++j) {
          uint32_t vertex_index = mesh.indices[i * edge_count + j];
          triangles[i].position[j] = mesh.vertices[vertex_index].position;
          triangles[i].normal[j] = mesh.vertices[vertex_index].normal;
          triangles[i].uv[j] = mesh.vertices[vertex_index].uv;
          triangles[i].vertex_indices[j] = vertex_index;
        }
      }

      return triangles;
    }

      // ---------------------model hierarchy---------------------


      // ---------------------forward kinematics---------------------

      using anim_key_t = std::string;

      uint32_t fk_set_rot(const anim_key_t& key, const fan::string& bone_id,
        f32_t dt,
        const fan::quat& quat
      ) {
        fan::vec3 axis;
        f32_t angle;
        quat.to_axis_angle(axis, angle);
        return fk_set_rot(key, bone_id, dt, axis, angle);
      }

      uint32_t fk_set_rot(const anim_key_t& key, const fan::string& bone_id,
        f32_t dt,
        const fan::vec3& axis,
        f32_t angle
      ) {
        auto& node = animation_list[key];
        auto found = node.bone_transforms.find(bone_id);
        if (found == node.bone_transforms.end()) {
          fan::throw_error("could not find bone:" + bone_id);
        }
        auto& transform = found->second;

        f32_t target = dt * 1000.f;
        auto it = std::upper_bound(transform.position_timestamps.begin(), transform.position_timestamps.end(), target);

        int insert_pos = std::distance(transform.position_timestamps.begin(), it);

        transform.position_timestamps.insert(transform.position_timestamps.begin() + insert_pos, dt * 1000.f);
        transform.rotation_timestamps.insert(transform.rotation_timestamps.begin() + insert_pos, dt * 1000.f);
        transform.scale_timestamps.insert(transform.scale_timestamps.begin() + insert_pos, dt * 1000.f);
        transform.positions.insert(transform.positions.begin() + insert_pos, 0);
        transform.rotations.insert(transform.rotations.begin() + insert_pos, fan::quat::from_axis_angle(axis, angle));
        transform.scales.insert(transform.scales.begin() + insert_pos, 1);

        return insert_pos;
      }

      static bool fk_get_time_fraction(const std::vector<f32_t>& times, f32_t dt, std::pair<uint32_t, f32_t>& fp) {
        if (times.empty()) {
          return true;
        }
        auto it = std::upper_bound(times.begin(), times.end(), dt);
        uint32_t segment = std::distance(times.begin(), it);
        if (times.size() == 1) {
          if (times[0] == 0) {
            fp = { 0, 1.f };
          }
          else {
            fp = { 0,  std::clamp((dt) / (times[0]), 0.0f, 1.0f) };
          }
          return false;
        }
        if (segment == 0) {
          segment++;
        }

        segment = fan::clamp(segment, uint32_t(0), uint32_t(times.size() - 1));

        f32_t start = times[segment - 1];
        f32_t end = times[segment];
        // clamping with default_animation will make it have cut effect
        fp = { segment, std::clamp((dt - start) / (end - start), 0.0f, 1.0f) };
        return false;
      }
      bool fk_parse_bone_data(const bone_transform_track_t& btt, fan::vec3& position, fan::quat& rotation, fan::vec3& scale) {
        std::pair<uint32_t, f32_t> fp;
        {// position
          if (fk_get_time_fraction(btt.position_timestamps, dt, fp)) {
            return true;
          }
          if (fp.first != (uint32_t)-1) {
            if (fp.first == 0) {
              position = fan::mix(fan::vec3(0), btt.positions[0], fp.second);
            }
            else {
              fan::vec3 p1 = btt.positions[fp.first - 1];
              fan::vec3 p2 = btt.positions[fp.first];
              fan::vec3 pos = fan::mix(p1, p2, fp.second);
              // +=
              position = pos * btt.weight;
            }
          }
        }

        {// rotation
          if (fk_get_time_fraction(btt.rotation_timestamps, dt, fp)) {
            return true;
          }
          if (fp.first != (uint32_t)-1) {
            if (fp.first == 0) {
              rotation = fan::quat::slerp(fan::quat(), btt.rotations[0], fp.second);
            }
            else {
              fan::quat rotation1 = btt.rotations[fp.first - 1];
              fan::quat rotation2 = btt.rotations[fp.first];

              fan::quat rot = fan::quat::slerp(rotation1, rotation2, fp.second);
              rotation = rot;
            }
          }
        }

        { // size
          if (fk_get_time_fraction(btt.scale_timestamps, dt, fp)) {
            return true;
          }
          if (fp.first != (uint32_t)-1) {
            if (fp.first == 0) {
              scale = fan::mix(fan::vec3(1), btt.scales[0], fp.second);
            }
            else {
              fan::vec3 s1 = btt.scales[fp.first - 1];
              fan::vec3 s2 = btt.scales[fp.first];
              // +=
              scale = fan::mix(s1, s2, fp.second) * btt.weight;
            }
          }
        }
        return false;
      }
      void fk_get_pose(animation_data_t& animation, bone_t& bone) {

        // this fmods other animations too. dt per animation? or with weights or max of all animations?
        if (animation.duration != 0) {
          dt = fmod(dt, longest_animation);
        }
        else {
          //dt = 0;
        }
        fan::vec3 position = 0, scale = 1;
        fan::quat rotation;

        bool not_enough_info = false;

        auto found = animation.bone_transforms.find(bone.name);
        if (found == animation.bone_transforms.end()) {
          int i = 0;
          if (animation.bone_poses.empty()) {
            animation.bone_poses.resize(bone_count, { .position{}, .rotation{}, .scale = 1 });
          }
          for (const auto& bt : animation.bone_transforms) {
            not_enough_info = fk_parse_bone_data(bt.second, position, rotation, scale);
           /* if (not_enough_info) {
              animation.bone_poses[bone.id].position = fan::vec3(0);
              animation.bone_poses[bone.id].rotation = fan::quat();
              animation.bone_poses[bone.id].scale = 1;
            }
            else {
              animation.bone_poses[bone.id].position = position;
              animation.bone_poses[bone.id].scale = scale;
              animation.bone_poses[bone.id].rotation = rotation;
            }*/
              animation.bone_poses[bone.id].position = fan::vec3(0);
              animation.bone_poses[bone.id].rotation = fan::quat();
              animation.bone_poses[bone.id].scale = 1;
            ++i;
          }
        }
        else {
         f32_t prev_weight = found->second.weight;
          found->second.weight = animation.weight;

          not_enough_info = fk_parse_bone_data(found->second, position, rotation, scale);
          //                          active_bone?
          /*if (showing_temp_rot && bone.id == -1) {
            rotation = temp_rotations[bone.id];
            not_enough_info = false;
          }*/

          found->second.weight = prev_weight;
          fan::mat4 local_transform{ 0 };
          if (not_enough_info) {
            local_transform = bone.transform;
           // local_transform = fan::mat4(1);
          }
          if (not_enough_info) {
            animation.bone_poses[bone.id].position = fan::vec3(0);
            animation.bone_poses[bone.id].rotation = fan::quat();
            animation.bone_poses[bone.id].scale = 1;
          }
          else {
            animation.bone_poses[bone.id].position = position;
            animation.bone_poses[bone.id].scale = scale;
            animation.bone_poses[bone.id].rotation = rotation;
          }
        }

        for (bone_t* child : bone.children) {
          fk_get_pose(animation, *child);
        }
      }
      void fk_interpolate_animations(std::vector<fan::mat4>& out_bone_transforms, bone_t& bone, fan::mat4& parent_transform) {
        fan::mat4 local_transform = bone.transform;

        bool has_active_animation = false;
        float total_weight = 0;

        for (const auto& [key, anim] : animation_list) {
          if (anim.weight > 0 && !anim.bone_poses.empty() && !anim.bone_transforms.empty()) {
            total_weight += anim.weight;
            has_active_animation = true;
          }
        }

        if (has_active_animation) {
          local_transform = bone.transform;

          for (const auto& [key, anim] : animation_list) {
            if (anim.weight > 0 && !anim.bone_poses.empty()) {
              float normalized_weight = anim.weight / total_weight;
              const auto& pose = anim.bone_poses[bone.id];

              fan::mat4 anim_transform;
              switch (anim.type) {
              case animation_data_t::type_e::nonlinear_animation: {
                anim_transform = fan::mat4(1);
                break;
              }
              case animation_data_t::type_e::custom: {
                anim_transform = bone.transform;
                break;
              }
              default: {
                fan::throw_error("invalid animation type");
                break;
              }
              }
              
              anim_transform = anim_transform.translate(pose.position);
              anim_transform = anim_transform * fan::mat4(pose.rotation);
              anim_transform = anim_transform.scale(pose.scale);
              for (int i = 0; i < fan::mat4::size(); i++) {
                local_transform[i] = local_transform[i] * (1.0f - normalized_weight) +
                  anim_transform[i] * normalized_weight;
              }
            }
          }
        }

        fan::mat4 global_transform = parent_transform * local_transform;
        out_bone_transforms[bone.id] = global_transform * bone.offset;
        bone.global_transform = global_transform * bone.offset;

        for (bone_t* child : bone.children) {
          fk_interpolate_animations(out_bone_transforms, *child, global_transform);
        }
      }
      std::vector<fan::mat4> fk_calculate_transformations() {
        std::vector<fan::mat4> fk_transformations = bone_transforms;

        fk_interpolate_animations(fk_transformations, *root_bone, m_transform);

        if (p.use_cpu) {
          for (uint32_t i = 0; i < meshes.size(); ++i) {
            calculate_vertices(fk_transformations, i, fan::mat4(1));
          }
        }
        return fk_transformations;
      }
      void fk_calculate_poses() {
        for (auto& i : animation_list) {
          fk_get_pose(i.second, *root_bone);
        }
      }

      // ---------------------forward kinematics---------------------


      // ---------------------animation---------------------


      void load_animations() {
        for (uint32_t j = 0; j < scene->mNumAnimations; ++j) {
          aiAnimation* anim = scene->mAnimations[j];
          auto& animation = animation_list[anim->mName.C_Str()];
          if (active_anim.empty()) {
            active_anim = anim->mName.C_Str();
          }
          animation.type = animation_data_t::type_e::nonlinear_animation;
          double ticksPerSecond = anim->mTicksPerSecond != 0 ? anim->mTicksPerSecond : 25.0;
    
          // Convert duration from ticks to seconds, then to milliseconds
          animation.duration = (anim->mDuration / ticksPerSecond) * 1000.0;
          longest_animation = std::max((f32_t)animation.duration, longest_animation);
          animation.weight = 0;
          for (int i = 0; i < anim->mNumChannels; i++) {
            aiNodeAnim* channel = anim->mChannels[i];
            fan_3d::model::bone_transform_track_t track;
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
      }

      anim_key_t active_anim;
      std::unordered_map<anim_key_t, animation_data_t> animation_list;
      f32_t longest_animation = 0;

      fan::string create_an(const fan::string& key, f32_t weight = 1.f, f32_t duration = 1.f) {
        if (animation_list.empty()) {
          if (active_anim.empty()) {
            active_anim = key;
          }
        }
        auto& node = animation_list[key];
        node.weight = weight;
        //node = m_animation;
        node.duration = duration * 1000.f;
        node.type = animation_data_t::type_e::custom;
        longest_animation = std::max((f32_t)node.duration, longest_animation);
        // initialize with tpose
        node.bone_poses.resize(bone_count, { .position{}, .rotation{}, .scale = 1 });
        iterate_bones(*root_bone,
          [&](bone_t& bone) {
            node.bone_transforms[bone.name];
          }
        );

        // user should worry about it
        //for (auto& anim : animation_list) {
        //  anim.second.weight = 1.f / animation_list.size();
        //}
        return key;
      }

      uint32_t get_active_animation_id() {
        if (active_anim.empty()) {
          fan::throw_error("no active animation");
        }
        auto found = animation_list.find(active_anim);
        if (found == animation_list.end()) {
          fan::throw_error("trying to access invalid animation:" + active_anim);
        }

        return std::distance(animation_list.begin(), found);
      }
      animation_data_t& get_active_animation() {
        if (active_anim.empty()) {
          fan::throw_error("no active animation");
        }
        auto found = animation_list.find(active_anim);
        if (found == animation_list.end()) {
          fan::throw_error("trying to access invalid animation:" + active_anim);
        }
        // TODO might be illegal if its temporary var
        return found->second;
      }


      // ---------------------animation---------------------


      // ---------------------gui---------------------

      void print_bone_recursive(bone_t* bone, int depth = 0) {
        if (!bone) {
          return;
        }

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_Framed;
        if (bone->children.empty()) {
          flags |= ImGuiTreeNodeFlags_Leaf;
        }

        if (depth > 0) {
          ImGui::Indent(10.0f);
        }

        bool node_open = ImGui::TreeNodeEx(bone->name.c_str(), flags);

        if (node_open) {
          ImGui::PushID(bone);

          ImGui::BeginGroup();
    
          f32_t speed = 1.0f;
          ImGui::PushItemWidth(150.0f);
    
          ImGui::Text("Rotation");
          ImGui::SameLine();
          if (ImGui::Button("Reset")) {
            bone->rotation = 0;
          }
          ImGui::SliderFloat("X", &bone->rotation.x, -fan::math::pi, fan::math::pi, "%.3f", speed);
          ImGui::SliderFloat("Y", &bone->rotation.y, -fan::math::pi, fan::math::pi, "%.3f", speed);
          ImGui::SliderFloat("Z", &bone->rotation.z, -fan::math::pi, fan::math::pi, "%.3f", speed);

          ImGui::PopItemWidth();
          ImGui::EndGroup();

          ImGui::PopID();

          // Print children
          for (bone_t* child : bone->children) {
            print_bone_recursive(child, depth + 1);
          }
          ImGui::TreePop();
        }

        if (depth > 0) {
          ImGui::Unindent(10.0f);
        }
      }
      void print_bones(bone_t* root_bone) {
        if (!root_bone) {
          ImGui::TextDisabled("No skeleton loaded");
          return;
        }

        ImGui::Begin("Bone Hierarchy");
  
        if (ImGui::Button("Reset All Rotations")) {
          std::function<void(bone_t*)> reset_rotations = [&](bone_t* bone) {
            if (bone) {
              bone->rotation = 0;
              for (bone_t* child : bone->children) {
                reset_rotations(child);
              }
            }
          };
          reset_rotations(root_bone);
        }

        ImGui::Separator();
        print_bone_recursive(root_bone);
        ImGui::End();
      }
      void display_animations() {
        for (auto& anim_pair : animation_list) {
          bool nodeOpen = ImGui::TreeNode(anim_pair.first.c_str());
          if (nodeOpen) {
            auto& anim = anim_pair.second;
            bool time_stamps_open = ImGui::TreeNode("timestamps");
            if (time_stamps_open) {
              iterate_bones(*root_bone, [&](fan_3d::model::bone_t& bone) {
                auto& bt = anim.bone_transforms[bone.name];
                uint32_t data_count = bt.rotation_timestamps.size();
                if (data_count) {
                  bool node_open = ImGui::TreeNode(bone.name.c_str());
                  if (node_open) {
                    for (int i = 0; i < bt.rotation_timestamps.size(); ++i) {
                      ImGui::DragFloat(
                        ("rotation:" + std::to_string(i)).c_str(),
                        &anim.bone_transforms[bone.name].rotation_timestamps[i]
                      );
                    }
                    ImGui::TreePop();
                  }
                }
                });
              ImGui::TreePop();
            }
            bool properties_open = ImGui::TreeNode("properties");
            if (properties_open) {
              iterate_bones(*root_bone, [&](fan_3d::model::bone_t& bone) {
                auto& bt = anim.bone_transforms[bone.name];
                uint32_t data_count = bt.rotations.size();
                if (data_count) {
                  bool node_open = ImGui::TreeNode(bone.name.c_str());
                  if (node_open) {
                    for (int i = 0; i < bt.rotations.size(); ++i) {
                      ImGui::DragFloat4(
                        ("rotation:" + std::to_string(i)).c_str(),
                        anim.bone_transforms[bone.name].rotations[i].data()
                      );
                    }
                    ImGui::TreePop();
                  }
                }
              });
              ImGui::TreePop();
            }
            ImGui::TreePop();
          }
        }
      }

      void mouse_modify_joint() {
        static constexpr f64_t delta_time_divier = 1e+7;
        f32_t duration = 1.f;
        ImGui::DragFloat("current time", &dt, 0.1, 0, fmod(duration, std::max(longest_animation, 1.f)));
        //ImGui::Text(fan::format("camera pos: {}\ntotal time: {:.2f}", gloco->default_camera_3d->camera.position, fms.default_animation.duration).c_str());
        static bool play = false;
        if (ImGui::Checkbox("play animation", &play)) {
          showing_temp_rot = false;
        }
        static int x = 0;
        static fan::time::clock c;
        if (play) {
          if (x == 0) {
            c.start();
            x++;
          }
          dt = c.elapsed() / delta_time_divier;
        }


        int current_id = get_active_animation_id();
        std::vector<const char*> animations;
        for (auto& i : animation_list) {
          animations.push_back(i.first.c_str());
        }
        if (ImGui::ListBox("animation list", &current_id, animations.data(), animations.size())) {
          active_anim = animations[current_id];
        }
        ImGui::DragFloat("animation weight", &animation_list[active_anim].weight, 0.01, 0, 1);

        if (active_bone != -1) {
          auto& anim = get_active_animation();
          auto& bt = anim.bone_transforms[bone_strings[active_bone]];
          for (int i = 0; i < bt.rotations.size(); ++i) {
            ImGui::DragFloat4(("rotations:" + std::to_string(i)).c_str(), bt.rotations[i].data(), 0.01);
          }

          static int32_t current_frame = 0;
          if (!play) {
            dt = current_frame;
          }
          else {
            current_frame = fmodf(c.elapsed() / delta_time_divier, anim.duration);
          }

          static f32_t prev_frame = 0;
          if (prev_frame != dt) {
            showing_temp_rot = false;
            prev_frame = dt;
          }

          if (ImGui::Button("save keyframe")) {
            fk_set_rot(active_anim, bone_strings[active_bone], current_frame / 1000.f, anim.bone_poses[active_bone].rotation);
          }

          //fan::print(current_frame);
          int32_t startFrame = 0;
          int32_t endFrame = std::ceil(duration);
          if (ImGui::BeginNeoSequencer("Sequencer", &current_frame, &startFrame, &endFrame, fan::vec2(0),
            ImGuiNeoSequencerFlags_EnableSelection |
            ImGuiNeoSequencerFlags_Selection_EnableDragging |
            ImGuiNeoSequencerFlags_Selection_EnableDeletion |
            ImGuiNeoSequencerFlags_AllowLengthChanging)) {
            static bool transform_open = true;
            if (ImGui::BeginNeoGroup("Transform", &transform_open))
            {

              if (ImGui::BeginNeoTimelineEx("rotation"))
              {
                if (bt.rotation_timestamps.size()) {
                  for (int i = 0; i < bt.rotation_timestamps.size(); ++i) {
                    int32_t p = bt.rotation_timestamps[i];
                    ImGui::NeoKeyframe(&p);
                    bt.rotation_timestamps[i] = p;
                    //ImGui::DragFloat(("timestamps:" + std::to_string(i)).c_str(), &bt.rotation_timestamps[i], 1);
                  }
                }
                // delete
                if (false)
                {
                  uint32_t count = ImGui::GetNeoKeyframeSelectionSize();

                  ImGui::FrameIndexType* toRemove = new ImGui::FrameIndexType[count];

                  ImGui::GetNeoKeyframeSelection(toRemove);

                  //Delete keyframes from your structure
                }
                ImGui::EndNeoTimeLine();
                ImGui::EndNeoGroup();
              }
            }

            ImGui::EndNeoSequencer();
          }
        }
      }

      int active_bone = -1;
      bool toggle_rotate = false;
      bool showing_temp_rot = false;
      
      // ---------------------gui---------------------

      const aiScene* scene;
      bone_t* root_bone = nullptr;

      std::vector<pm_material_data_t> material_data_vector;
      std::vector<fan::mat4> bone_transforms;
      std::vector<mesh_t> meshes;
      std::vector<mesh_t> calculated_meshes;
      std::unordered_map<std::string, bone_t*> bone_map;

      std::vector<fan::string> bone_strings;

      fms_model_info_t p;

      fan::mat4 m_transform{1};
      uint32_t bone_count = 0;
      f32_t dt = 0;
    };
  }
}