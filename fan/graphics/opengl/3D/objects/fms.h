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
      struct joint_pose_t {
        fan::vec3 position = 0;
        fan::quat rotation;
        fan::vec3 scale = 1;
      };
      std::vector<joint_pose_t> joint_poses;
      std::vector<fan::mat4> pose;
      f32_t weight = 1;
    };

    struct bone_t {
      std::string name;
      fan::mat4 offset;
      fan::mat4 transform;
      int id = -1;
      std::vector<bone_t*> children;
      bone_t* parent;

      // this needs to be user mat4
      float rotationX = 0.0f;
      float rotationY = 0.0f;
      float rotationZ = 0.0f;
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
            float weight = bone->mWeights[j].mWeight;

            // find the slot with minimum weight and replace if current weight is larger
            int min_index = 0;
            float min_weight = temp_vertices[vertexId].bone_weights[0];

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
          float sum = vertex.bone_weights.x + vertex.bone_weights.y +
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
          rootBone = bone;
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
      void update_bone_rotation(const std::string& boneName, float x, float y, float z) {
        auto it = bone_map.find(boneName);
        if (it != bone_map.end()) {
          bone_t* bone = it->second;
          bone->rotationX = x;
          bone->rotationY = y;
          bone->rotationZ = z;
        }
      }
      void update_bone_transforms_impl(bone_t* bone, const fan::mat4& parentTransform) {
        if (!bone) return;


        fan::mat4 translation(1);
        fan::mat4 rotation = fan::mat4(1.0f);
        fan::mat4 scale(1);
        rotation = rotation.rotate(fan::math::radians(bone->rotationX), fan::vec3(1, 0, 0));
        rotation = rotation.rotate(fan::math::radians(bone->rotationY), fan::vec3(0, 1, 0));
        rotation = rotation.rotate(fan::math::radians(bone->rotationZ), fan::vec3(0, 0, 1));

        fan::mat4 node_transform = bone->transform;

        fan::mat4 local_transform = translation * rotation * scale;
        fan::mat4 globalTransform = parentTransform * node_transform * local_transform;

        bone_transforms[bone->id] = globalTransform  * bone->offset;

        for (bone_t* child : bone->children) {
          update_bone_transforms_impl(child, globalTransform);
        }
      }
      void update_bone_transforms() {
        bone_transforms.clear();
        bone_transforms.resize(bone_count);
        if (rootBone) {
          update_bone_transforms_impl(rootBone, m_transform);
        }
      }
      fan::vec4 calculate_bone_transform(uint32_t mesh_id, uint32_t vertex_id) {
        auto& vertex = meshes[mesh_id].vertices[vertex_id];

        fan::vec4 totalPosition(0.0);

        for (int i = 0; i < 4; i++) {
          float weight = vertex.bone_weights[i];
          int boneId = vertex.bone_ids[i];
          if (boneId == -1) {
            continue;
          }

          if (boneId >= bone_transforms.size()) {
            totalPosition = fan::vec4(vertex.position, 1.0);
            break;
          }
          fan::vec4 local_position = bone_transforms[boneId] * fan::vec4(vertex.position, 1.0);
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

      void calculate_modified_vertices(uint32_t mesh_id, const fan::mat4& model = fan::mat4(1)) {

        update_bone_transforms();

        for (int i = 0; i < meshes[mesh_id].vertices.size(); ++i) {

          fan::vec3 v = meshes[mesh_id].vertices[i].position;
          if (bone_transforms.empty()) {
            fan::vec4 vertex_position = m_transform * model * fan::vec4(v, 1.0);
            calculated_meshes[mesh_id].vertices[i].position = fan::vec3(vertex_position.x, vertex_position.y, vertex_position.z);
          }
          else {
            fan::vec4 interpolated_bone_transform = calculate_bone_transform(mesh_id, i);

            fan::vec4 vertex_position = fan::vec4(v, 1.0);

            fan::vec4 result = interpolated_bone_transform;

            calculated_meshes[mesh_id].vertices[i].position = fan::vec3(result.x, result.y, result.z);
          }
        }
      }

      // triangle consists of 3 vertices
      struct one_triangle_t {
        fan::vec3 position[3]{};
        fan::vec3 normal[3]{};
        fan::vec2 uv[3]{};
      };

      void get_triangle_vec(uint32_t mesh_id, std::vector<one_triangle_t>* triangles) {
        static constexpr int edge_count = 3;
        // ignore i for now since only one scene
        triangles->resize(calculated_meshes[mesh_id].vertices.size() / edge_count);
        for (int i = 0; i < calculated_meshes[mesh_id].vertices.size(); ++i) {
          (*triangles)[i / edge_count].position[i % edge_count] = calculated_meshes[mesh_id].vertices[i].position;
          (*triangles)[i / edge_count].uv[i % edge_count] = calculated_meshes[mesh_id].vertices[i].uv;
        }
      }

      using anim_key_t = std::string;

      anim_key_t active_anim;
      std::unordered_map<anim_key_t, animation_data_t> animation_list;

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

        float target = dt * 1000.f;
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
    
          float speed = 1.0f;
          ImGui::PushItemWidth(150.0f);
    
          ImGui::Text("Rotation");
          ImGui::SameLine();
          if (ImGui::Button("Reset")) {
            bone->rotationX = 0.0f;
            bone->rotationY = 0.0f;
            bone->rotationZ = 0.0f;
          }

          ImGui::SliderFloat("X", &bone->rotationX, -180.0f, 180.0f, "%.1f°", speed);
          ImGui::SliderFloat("Y", &bone->rotationY, -180.0f, 180.0f, "%.1f°", speed);
          ImGui::SliderFloat("Z", &bone->rotationZ, -180.0f, 180.0f, "%.1f°", speed);
    
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
      void print_bones(bone_t* rootBone) {
        if (!rootBone) {
          ImGui::TextDisabled("No skeleton loaded");
          return;
        }

        ImGui::Begin("Bone Hierarchy");
  
        if (ImGui::Button("Reset All Rotations")) {
          std::function<void(bone_t*)> reset_rotations = [&](bone_t* bone) {
            if (bone) {
              bone->rotationX = 0.0f;
              bone->rotationY = 0.0f;
              bone->rotationZ = 0.0f;
              for (bone_t* child : bone->children) {
                reset_rotations(child);
              }
            }
          };
          reset_rotations(rootBone);
        }

        ImGui::Separator();
        print_bone_recursive(rootBone);
        ImGui::End();
      }

      const aiScene* scene;
      bone_t* rootBone = nullptr;

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