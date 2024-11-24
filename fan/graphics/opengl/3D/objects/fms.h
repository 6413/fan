#pragma once

namespace fan_3d {
  namespace model {
    using namespace fan::opengl;

    struct vertex_t {
      fan::vec3 position;
      fan::vec3 normal;
      fan::vec2 uv;
      fan::vec4 bone_ids;
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
      f_t duration = 10;
      // ticks per second
      f_t tps = 16;
      std::unordered_map<std::string, bone_transform_track_t> bone_transforms;
      struct bone_pose_t {
        fan::vec3 position = 0;
        fan::quat rotation;
        fan::vec3 scale = 1;
      };
      std::vector<bone_pose_t> bone_poses;
      std::vector<fan::mat4> pose;
      f32_t weight = 1;
    };

    struct bone_t {
      int id = -1;
      std::string name;
      fan::mat4 offset;
      fan::mat4 transform;
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
      std::string texture_id[AI_TEXTURE_TYPE_MAX + 1];
      fan::vec4 color[AI_TEXTURE_TYPE_MAX + 1];
    };

    struct fms_model_info_t {
      fan::string path;
      std::string texture_path;
    };

    // fan model stuff
    struct fms_t {
      fms_t() = default;
      fms_t(const fms_model_info_t& fmi) {
        if (!load_model(fmi.path)) {
          fan::throw_error("failed to load model:" + fmi.path);
        }
        this->scene = scene;
        calculated_meshes = meshes;
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
            fan::string file_path = texture_path + "/" + scene->GetShortFilename(path.C_Str());
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
          material_data.texture_id[i] = std::string("");
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

        meshes.clear();

        for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
          mesh_t mesh = process_mesh(scene->mMeshes[i]);
          aiMesh* ai_mesh = scene->mMeshes[i];
          pm_material_data_t material_data = load_materials(ai_mesh);
          if (std::string tret = load_textures(mesh, ai_mesh); !tret.empty()) {
            material_data.texture_id[i] = tret;
          }
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
            fan::vec4 result = interpolated_bone_transform;

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
      };

      std::vector<one_triangle_t> get_triangles(uint32_t mesh_id) {
        std::vector<one_triangle_t> triangles;
        static constexpr int edge_count = 3;
        const auto& mesh = calculated_meshes[mesh_id];
        // ignore i for now since only one scene
        triangles.resize(mesh.indices.size() / edge_count);
        for (uint32_t i = 0; i < calculated_meshes[mesh_id].vertices.size(); ++i) {
          for (int j = 0; j < edge_count; ++j) {
            uint32_t vertex_index = mesh.indices[i + j];
            triangles[i / edge_count].position[j] = mesh.vertices[vertex_index].position;
            triangles[i / edge_count].normal[j] = mesh.vertices[vertex_index].normal;
            triangles[i / edge_count].uv[j] = mesh.vertices[vertex_index].uv;
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

      void fk_get_pose(animation_data_t& animation, bone_t& bone, fan::mat4& parent_transform) {

        dt = fmod(dt, animation.duration);
        fan::vec3 position = 0, scale = 1;
        fan::quat rotation;

        bool not_enough_info = false;

        auto found = animation.bone_transforms.find(bone.name);
        f32_t prev_weight = found->second.weight;
        if (found == animation.bone_transforms.end()) {
          fan::throw_error("invalid bone data");
        }
        else {
          found->second.weight = 1.f;

          not_enough_info = fk_parse_bone_data(found->second, position, rotation, scale);
          //                          active_bone?
          /*if (showing_temp_rot && bone.id == -1) {
            rotation = temp_rotations[bone.id];
            not_enough_info = false;
          }*/

          found->second.weight = prev_weight;
        }

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

        fan::mat4 global_transform = parent_transform * local_transform;

        for (bone_t* child : bone.children) {
          fk_get_pose(animation, *child, global_transform);
        }
      }

      void fk_interpolate_animations(std::vector<fan::mat4>& out_bone_transforms, bone_t& bone, fan::mat4& parent_transform) {

        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;

        fan::mat4 global_transform;

        if (animation_list.empty()) {
          // 
          out_bone_transforms[bone.id] = parent_transform * bone.offset;
        }
        else {
          for (auto& apair : animation_list) {
            auto& a = apair.second;
            position += a.bone_poses[bone.id].position * a.weight;
            fan::quat slerped_quat = fan::quat::slerp(fan::quat(1, 0, 0, 0), a.bone_poses[bone.id].rotation, a.weight);
            rotation = (rotation * slerped_quat).normalize();
            scale += a.bone_poses[bone.id].scale * a.weight;
          }

          // ? 
//          fan::mat4 local_transform = bone.local_transform;
          fan::mat4 local_transform = bone.transform;
          local_transform = local_transform.translate(position);
          local_transform = local_transform.rotate(rotation);
          local_transform = local_transform.scale(scale);
          out_bone_transforms[bone.id] = m_transform * parent_transform * local_transform * bone.offset;
          global_transform = parent_transform * local_transform;
        }

        for (bone_t* child : bone.children) {
          fk_interpolate_animations(out_bone_transforms, *child, global_transform);
        }
      }

      void fk_calculate_transformations(const fan::mat4& model = fan::mat4(1)) {
        std::vector<fan::mat4> fk_transformations = bone_transforms;

        fk_interpolate_animations(fk_transformations, *root_bone, m_transform);

        for (uint32_t i = 0; i < meshes.size(); ++i) {
          calculate_vertices(fk_transformations, i, model);
        }
      }

      void fk_calculate_poses() {
        fan::mat4 initial{1};
        for (auto& i : animation_list) {
          fk_get_pose(i.second, *root_bone, initial);
        }
      }

      // ---------------------forward kinematics---------------------


      // ---------------------animation---------------------


      anim_key_t active_anim;
      std::unordered_map<anim_key_t, animation_data_t> animation_list;

      fan::string create_an(const fan::string& key, f32_t weight = 1.f, f32_t duration = 5.f) {
        if (animation_list.empty()) {
          active_anim = key;
        }
        auto& node = animation_list[key];
        node.weight = weight;
        //node = m_animation;
        node.duration = duration;
        // initialize with tpose
        node.bone_poses.resize(bone_count, { {}, {}, 1 });
        node.pose.resize(bone_count, fan::mat4(1));
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
          ImGui::SliderFloat("X", &bone->rotation.x, 0, fan::math::pi * 2, "%.3f", speed);
          ImGui::SliderFloat("Y", &bone->rotation.y, 0, fan::math::pi * 2, "%.3f", speed);
          ImGui::SliderFloat("Z", &bone->rotation.z, 0, fan::math::pi * 2, "%.3f", speed);

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

      // ---------------------gui---------------------

      const aiScene* scene;
      bone_t* root_bone = nullptr;

      std::vector<pm_material_data_t> material_data_vector;
      std::vector<fan::mat4> bone_transforms;
      std::vector<mesh_t> meshes;
      std::vector<mesh_t> calculated_meshes;
      std::unordered_map<std::string, bone_t*> bone_map;

      std::string texture_path;

      fan::mat4 m_transform{1};
      uint32_t bone_count = 0;
      f32_t dt = 0;
    };
  }
}