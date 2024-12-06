#pragma once

#include <assimp/Exporter.hpp>
#include <locale>

#include <fan/types/vector.h>
#include <fan/types/matrix.h>
#include <fan/types/quaternion.h>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include <fan/types/dme.h>

#include <fan/graphics/image_load.h>

namespace fan_3d {
  namespace model {
    struct vertex_t {
      fan::vec3 position;
      fan::vec3 normal;
      fan::vec2 uv;
      fan::vec4i bone_ids;
      fan::vec4 bone_weights;
      fan::vec3 tangent;
      fan::vec3 bitangent;
      fan::vec4 color;
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
        fan::quat rotation = fan::quat(-1234, 0, 0, 0);
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
      std::vector<bone_transform_track_t> bone_transform_tracks;
      std::string name;
    };
    struct bone_t {
      int id = -1; // also depth
      std::string name;
      fan::mat4 offset;
      fan::mat4 transformation;
      fan::mat4 world_matrix;
      fan::mat4 inverse_parent_matrix;
      fan::mat4 bone_transform; // for delta
      bone_t* parent;
      std::vector<bone_t*> children;
      fan::vec3 position = 0;
      fan::quat rotation;
      fan::vec3 scale = 1;
      
      fan::vec3 user_position = 0;
      fan::vec3 user_rotation = 0;
      fan::vec3 user_scale = 1;

      // this appears to be different than transformation
      fan::mat4 get_local_matrix() const {
        return fan::mat4(1).translate(position) * fan::mat4(1).rotate(rotation) * fan::mat4(1).scale(scale);
      }
    };
    struct mesh_t {
      std::vector<fan_3d::model::vertex_t> vertices;
      std::vector<uint32_t> indices;

      std::string texture_names[AI_TEXTURE_TYPE_MAX + 1]{};

#ifdef fms_use_opengl
      fan::opengl::core::vao_t vao;
      fan::opengl::core::vbo_t vbo;
      fan::opengl::GLuint ebo;
#endif
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
    // fan model stuff
    struct fms_t {
      struct properties_t {
        fan::string path;
        std::string texture_path;
        int use_cpu = false;
      };
      fms_t() = default;
      fms_t(const properties_t& fmi) {
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
          if (mesh->HasVertexColors(0)) {
            vertex.color = fan::vec4(
              mesh->mColors[0][i].r,
              mesh->mColors[0][i].g,
              mesh->mColors[0][i].b,
              mesh->mColors[0][i].a
            );
          }
          else {
            vertex.color = fan::vec4(1.0f);
          }
        }

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
            if (data == nullptr) {
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
              if (fan::image::load(file_path, &ii)) {
                return "";
              }
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
        for (uint32_t i = 0; i <= AI_TEXTURE_TYPE_MAX; i++) {
          material_data.color[i] = fan::vec4(1, 1, 1, 1);
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
        aiVector3D pos, scale;
        aiQuaternion rot;
        node->mTransformation.Decompose(scale, rot, pos);
        bone->position = pos;
        bone->scale = scale;
        bone->rotation = rot;
        bone->transformation = node->mTransformation;
        bone->world_matrix = get_world_matrix(bone, bone->get_local_matrix());
        bone->inverse_parent_matrix = get_inverse_parent_matrix(bone);
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
            if (root_bone->id > it->second->id) {
              root_bone = it->second;
            }
            bones.push_back(it->second);
            it->second->offset = bone->mOffsetMatrix;
          }
        }
      }
      void update_bone_rotation(const std::string& boneName, const fan::quat& rotation) {
        auto it = bone_map.find(boneName);
        if (it != bone_map.end()) {
          bone_t* bone = it->second;
          bone->rotation = rotation;
        }
      }
      void update_bone_transforms_impl(bone_t* bone, const fan::mat4& parentTransform) {
        if (!bone) return;

        fan::mat4 node_transform = bone->transformation;

        fan::mat4 globalTransform =
          parentTransform *
          node_transform
          ;

        bone_transforms[bone->id] = globalTransform * bone->offset;

        for (bone_t* child : bone->children) {
          update_bone_transforms_impl(child, globalTransform);
        }
      }
      void update_bone_transforms() {
        bone_transforms.clear();
        bone_transforms.resize(bone_count, fan::mat4(1));
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
        importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
        scene = (aiScene*)importer.ReadFile(path, 
          aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes | aiProcess_Triangulate | 
          aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | 
          aiProcess_JoinIdenticalVertices | aiProcess_GenBoundingBoxes);

        //scene = importer.GetOrphanedScene();
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
          fan::print(importer.GetErrorString());
          return false;
        }

        m_transform = scene->mRootNode->mTransformation;

        bone_count = 0;
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
          fan::vec3 min = ai_mesh->mAABB.mMin;
          fan::vec3 max = ai_mesh->mAABB.mMax;
          scale_divider = std::max(scale_divider, (max-min).abs().length());
        }
        update_bone_transforms();
        m_transform = m_transform.scale(1.0 / (scale_divider / 3));
        return true;
      }
      void calculate_vertices(const std::vector<fan::mat4>& bt, uint32_t mesh_id, const fan::mat4& model) {
        for (int i = 0; i < meshes[mesh_id].indices.size(); ++i) {
          uint32_t vertex_index = meshes[mesh_id].indices[i];
          fan::vec3 v = meshes[mesh_id].vertices[vertex_index].position;
          if (bt.empty()) {
            fan::vec4 vertex_position = m_transform * model * fan::vec4(v, 1.0);
            calculated_meshes[mesh_id].vertices[vertex_index].position = fan::vec3(vertex_position.x, vertex_position.y, vertex_position.z);
          }
          else {
            fan::vec4 interpolated_bone_transform = calculate_bone_transform(bt, mesh_id, vertex_index);
            fan::vec4 vertex_position = fan::vec4(v, 1.0);
            fan::vec4 result = model * interpolated_bone_transform;
            calculated_meshes[mesh_id].vertices[vertex_index].position = fan::vec3(result.x, result.y, result.z);
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

      uint32_t fk_set_rot(const anim_key_t& key, const uint32_t bone_id,
        f32_t dt,
        const fan::quat& quat
      ) {
        fan::vec3 axis;
        f32_t angle;
        quat.to_axis_angle(axis, angle);
        return fk_set_rot(key, bone_id, dt, axis, angle);
      }

      uint32_t fk_set_rot(const anim_key_t& key, uint32_t bone_id,
        f32_t dt,
        const fan::vec3& axis,
        f32_t angle
      ) {
        auto& node = animation_list[key];
        
        auto& transform = node.bone_transform_tracks[bone_id];

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
      uint8_t interpolate_bone_transform(const bone_transform_track_t& btt, fan::vec3& position, fan::quat& rotation, fan::vec3& scale) {
        std::pair<uint32_t, f32_t> fp;
        uint8_t ret = false;
        {// position
          position = btt.positions[0];
          bool tr = fk_get_time_fraction(btt.position_timestamps, dt, fp);
          ret |= ((uint8_t)tr << 0);
          if (!tr && fp.first != (uint32_t)-1) {
            if (fp.first == 0) {
              position = fan::mix(fan::vec3(0), btt.positions[0], fp.second);
            }
            else {
              fan::vec3 p1 = btt.positions[fp.first - 1];
              fan::vec3 p2 = btt.positions[fp.first];
              position = fan::mix(p1, p2, fp.second) * btt.weight;
            }
          }
        }
        {// rotation
          rotation = btt.rotations[0];
          bool tr = fk_get_time_fraction(btt.rotation_timestamps, dt, fp);
          ret |= ((uint8_t)tr << 1);
          if (!tr && fp.first != (uint32_t)-1) {
            if (fp.first == 0) {
              rotation = fan::quat::slerp(fan::quat(), btt.rotations[0], fp.second);
            }
            else {
              fan::quat rotation1 = btt.rotations[fp.first - 1];
              fan::quat rotation2 = btt.rotations[fp.first];
              rotation = fan::quat::slerp(rotation1, rotation2, fp.second);
            }
          }
        }
        { // size
          scale = btt.scales[0];
          bool tr = fk_get_time_fraction(btt.scale_timestamps, dt, fp);
          ret |= ((uint8_t)tr << 2);
          if (!tr && fp.first != (uint32_t)-1) {
            if (fp.first == 0) {
              scale = fan::mix(fan::vec3(1), btt.scales[0], fp.second);
            }
            else {
              fan::vec3 s1 = btt.scales[fp.first - 1];
              fan::vec3 s2 = btt.scales[fp.first];
              scale = fan::mix(s1, s2, fp.second) * btt.weight;
            }
          }
        }
        return ret;
      }
      void fk_get_pose(animation_data_t& animation, const bone_t& bone) {
        // this fmods other animations too. dt per animation? or with weights or max of all animations?
        // fix xd
        if (animation.duration != 0) {
          if (animation.weight == 1 && get_active_animation_id() != -1 && animation.duration == get_active_animation().duration) {
            dt = fmod(dt, animation.duration);
          }
          else {
            dt = fmod(dt, 100000);
          }
        }
        fan::vec3 position = 0, scale = 1;
        fan::quat rotation;

        const auto& bt = animation.bone_transform_tracks[bone.id];
        animation.bone_poses[bone.id].rotation.w = -9999;
        if (bt.positions.empty()) {
          return;
        }
        // returns bit for successsion for position, rotation, scale
        uint8_t r = interpolate_bone_transform(bt, position, rotation, scale);
        animation.bone_poses[bone.id].position = position;
        animation.bone_poses[bone.id].rotation = rotation;
        animation.bone_poses[bone.id].scale = scale;
      }
      void fk_interpolate_animations(std::vector<fan::mat4>& out_bone_transforms, bone_t& bone, fan::mat4& parent_transform) {
        fan::mat4 local_transform = bone.transformation;

        bool has_active_animation = false;
        float total_weight = 0;

        for (const auto& [key, anim] : animation_list) {
          if (anim.weight > 0 && !anim.bone_poses.empty() && !anim.bone_transform_tracks.empty()) {
            total_weight += anim.weight;
            has_active_animation = true;
            break;
          }
        }

        if (has_active_animation) {
          for (const auto& [key, anim] : animation_list) {
            if (anim.weight > 0 && !anim.bone_poses.empty()) {
              float normalized_weight = anim.weight / total_weight;
              const auto& pose = anim.bone_poses[bone.id];
              if (pose.rotation.w == -9999) { // uninitalized
                local_transform = bone.transformation;
              }
              else {
                static fan::vec3 translation = 0;
                local_transform = (
                  fan::translation_matrix(pose.position) *
                  fan::rotation_quat_matrix(pose.rotation) *
                  fan::scaling_matrix(pose.scale)
                );
              }
            }
          }
        }

        fan::mat4 global_transform = 
          parent_transform * 
          (fan::translation_matrix(bone.user_position) *
          fan::rotation_quat_matrix(fan::quat::from_angles(bone.user_rotation)) *
          fan::scaling_matrix(bone.user_scale)) *
          local_transform;
        out_bone_transforms[bone.id] = global_transform * bone.offset;
        bone.bone_transform = global_transform;

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
          if (i.second.weight > 0) {
            for (auto bone : bones) {
              fk_get_pose(i.second, *bone);
            }
          }
        }
      }

      // ---------------------forward kinematics---------------------


      // ---------------------animation---------------------

      void load_animations() {
        for (uint32_t k = 0; k < scene->mNumAnimations; ++k) {
          aiAnimation* anim = scene->mAnimations[k];
          auto& animation = animation_list[anim->mName.C_Str()];
          animation.name = anim->mName.C_Str();
          if (active_anim.empty()) {
            active_anim = anim->mName.C_Str();
          }
          if (animation.bone_poses.empty()) {
            animation.bone_poses.resize(bone_count);
          }
          if (animation.bone_transform_tracks.empty()) {
            animation.bone_transform_tracks.resize(bone_count);
          }
          animation.type = animation_data_t::type_e::nonlinear_animation;
          double ticksPerSecond = anim->mTicksPerSecond != 0 ? anim->mTicksPerSecond : 25.0;
          animation.duration = (anim->mDuration / ticksPerSecond) * 1000.0;
          longest_animation = std::max((f32_t)animation.duration, longest_animation);
          animation.weight = 0;
          int bone_id = 0;
          for (int i = 0; i < anim->mNumChannels; i++) {
            aiNodeAnim* channel = anim->mChannels[i];
            auto found = bone_map.find(channel->mNodeName.C_Str());
            if (found == bone_map.end()) {
              fan::print("unmapped bone, skipping...");
              continue;
            }
            fan::print(found->first);
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
            animation.bone_transform_tracks[found->second->id] = track;
          }

        }
      }
      fan::string create_an(const fan::string& key, f32_t weight = 1.f, f32_t duration = 1.f) {
        if (animation_list.empty()) {
          if (active_anim.empty()) {
            active_anim = key;
          }
        }
        auto& node = animation_list[key];
        node.weight = weight;
        node.duration = duration * 1000.f;
        node.type = animation_data_t::type_e::custom;
        longest_animation = std::max((f32_t)node.duration, longest_animation);
        // initialize with tpose
        node.bone_poses.resize(bone_count);
        iterate_bones(*root_bone,
          [&](bone_t& bone) {
            
          }
        );
        return key;
      }
      uint32_t get_active_animation_id() {
        if (active_anim.empty()) {
          return -1;
        }
        auto found = animation_list.find(active_anim);
        if (found == animation_list.end()) {
          fan::throw_error("trying to access invalid animation:" + active_anim);
        }

        return std::distance(animation_list.begin(), found);
      }
      animation_data_t& get_active_animation() {
        uint32_t id = get_active_animation_id();
        if (id == -1) {
          fan::throw_error("no active animation");
        }
        auto it = animation_list.begin();
        std::advance(it, id);
        return it->second;
      }
      bool export_animation(
        const std::string& animation_name_to_export,
        const std::string path,
        const std::string& format = "gltf2"
      ) {
        if (scene->mNumAnimations == 0) {
          fan::print("model has no animations");
          return 0;
        }
        uint32_t animation_index = -1;
        for (uint32_t i = 0; i < scene->mNumAnimations; ++i) {
          if (scene->mAnimations[i]->mName.C_Str() == animation_name_to_export) {
            animation_index = i;
            break;
          }
        }

        if (animation_index == -1) {
          fan::print("failed to find animation:" + animation_name_to_export);
          return 0;
        }
        aiScene* new_scene = new aiScene();
        new_scene->mRootNode = scene->mRootNode;
        new_scene->mMeshes = scene->mMeshes;
        new_scene->mNumMeshes = scene->mNumMeshes;
        new_scene->mMaterials = scene->mMaterials;
        new_scene->mNumMaterials = scene->mNumMaterials;
        if (scene->mNumAnimations > 0) {
          new_scene->mAnimations = new aiAnimation * [1];
          new_scene->mAnimations[0] = scene->mAnimations[animation_index];
          new_scene->mNumAnimations = 1;
        }

        for (int i = 0; i < new_scene->mAnimations[0]->mNumChannels; ++i) {
          fan::print("exporting:", new_scene->mAnimations[0]->mChannels[i]->mNodeName.C_Str());
        }

        Assimp::Exporter exporter;
        aiReturn result = exporter.Export(new_scene, format.c_str(), path.c_str());
        delete[] new_scene->mAnimations;

        new_scene->mRootNode = nullptr;
        new_scene->mMeshes = nullptr;
        new_scene->mMaterials = nullptr;
        new_scene->mAnimations = nullptr;
        delete new_scene;

        if (result != aiReturn_SUCCESS) {
          fan::print("failed to export animation", exporter.GetErrorString());
          return 0;
        }

        return 1;
      }
      struct bone_names_default_t : __dme_inherit(bone_names_default_t, std::vector<std::string>){
        bone_names_default_t() {}
        #define d(name, ...) __dme(name) = {{#name, ##__VA_ARGS__}};
        d(hips, "Torso");
        d(right_up_leg, "Upper_Leg_R", "RightUpLeg", "Upper_Leg.R", "R_UpperLeg");
        d(lower_leg_r, "Lower_Leg.R", "Right_knee", "R_LowerLeg");
        d(right_foot, "Foot_R", "RightFoot", "Foot.R", "Right_ankle", "R_Foot");
        d(right_toe_base, "RightToeBase", "Right_toe", "R_ToeBase");
        d(right_toe_end);
        d(spine);
        d(spine1, "Chest", "UpperChest");
        d(spine2);
        d(neck);
        d(head);
        d(head_top_end);
        d(left_shoulder, "Upper_Arm_L", "L_Shoulder");
        d(left_arm, "L_UpperArm");
        d(left_fore_arm, "Lower_Arm_L", "Left_elbow", "L_LowerArm");
        d(left_hand, "Hand_L", "Left_wrist", "L_Hand");
        d(left_hand_thumb1, "Thumb0_L");
        d(left_hand_thumb2, "Thumb1_L");
        d(left_hand_thumb3, "Thumb2_L");
        d(left_hand_thumb4, "Thumb2Tip_L");
        d(left_hand_index1, "IndexFinger1_L");
        d(left_hand_index2, "IndexFinger2_L");
        d(left_hand_index3, "IndexFinger3_L");
        d(left_hand_index4, "IndexFinger3Tip_L");
        d(left_hand_middle1, "MiddleFinger1_L");
        d(left_hand_middle2, "MiddleFinger2_L");
        d(left_hand_middle3, "MiddleFinger3_L");
        d(left_hand_middle4, "MiddleFinger3Tip_L");
        d(left_hand_ring1, "RingFinger1_L");
        d(left_hand_ring2, "RingFinger2_L");
        d(left_hand_ring3, "RingFinger3_L");
        d(left_hand_ring4, "RingFinger3Tip_L");
        d(left_hand_pinky1, "LittleFinger1_L");
        d(left_hand_pinky2, "LittleFinger2_L");
        d(left_hand_pinky3, "LittleFinger3_L");
        d(left_hand_pinky4, "LittleFinger3Tip_L");
        d(right_shoulder, "Upper_Arm_R", "R_Shoulder");
        d(right_arm, "R_UpperArm");
        d(right_fore_arm, "Lower_Arm_R", "Right_elbow", "R_LowerArm");
        d(right_hand, "Hand_R", "Right_wrist", "R_Hand");
        d(right_hand_thumb1, "Thumb0_R");
        d(right_hand_thumb2, "Thumb1_R");
        d(right_hand_thumb3, "Thumb2_R");
        d(right_hand_thumb4, "Thumb2Tip_R");
        d(right_hand_index1, "IndexFinger1_R");
        d(right_hand_index2, "IndexFinger2_R");
        d(right_hand_index3, "IndexFinger3_R");
        d(right_hand_index4, "IndexFinger3Tip_R");
        d(right_hand_middle1, "MiddleFinger1_R");
        d(right_hand_middle2, "MiddleFinger2_R");
        d(right_hand_middle3, "MiddleFinger3_R");
        d(right_hand_middle4, "MiddleFinger3Tip_R");
        d(right_hand_ring1, "RingFinger1_R");
        d(right_hand_ring2, "RingFinger2_R");
        d(right_hand_ring3, "RingFinger3_R");
        d(right_hand_ring4, "RingFinger3Tip_R");
        d(right_hand_pinky1, "LittleFinger1_R");
        d(right_hand_pinky2, "LittleFinger2_R");
        d(right_hand_pinky3, "LittleFinger3_R");
        d(right_hand_pinky4, "LittleFinger3Tip_R");
        d(left_up_leg, "Upper_Leg_L", "LeftUpLeg", "Upper_Leg.L", "L_UpperLeg");
        d(lower_leg_l, "Lower_Leg.L", "Left_knee", "L_LowerLeg");
        d(left_foot, "Foot_L", "LeftFoot", "Foot.L", "Left_ankle", "L_Foot");
        d(left_toe_base, "LeftToeBase", "Left_toe", "L_ToeBase");
        d(left_toe_end);
        #undef d

        uintptr_t size() {
          return this->GetMemberAmount();
        }
        auto& operator[](uintptr_t i) {
          return *this->NA(i);
        }
      };

      inline static bone_names_default_t bone_names_default;
      bone_names_default_t bone_names_anim;
      bone_names_default_t bone_names_model;
      void fancy_iterator(auto& iteration, auto func) {
        for (uintptr_t i = 0; i < iteration.size(); i++) {
          func(iteration[i]);
        }
      }
      uintptr_t get_bone_name_index(auto& iteration, std::string from) {
        uintptr_t longest_name_id = (uintptr_t)-1;
        uintptr_t longest_length = 0;
        uintptr_t shortest_length_from = (uintptr_t)-1;

        for (uintptr_t i = 0; i < iteration.size(); i++) {
          for (uintptr_t bni1 = 0; bni1 < iteration[i].size(); bni1++) {
            std::string bn = iteration[i][bni1];
            for (uintptr_t ip = 0; ip < from.size(); ip++) {
              uintptr_t bni = 0;
              auto nfrom = from.substr(ip);
              uintptr_t nfromi = 0;
              while (1) {
                if (bni == bn.size()) {
                  if (
                    (bni > longest_length) ||
                    (from.size() < shortest_length_from)
                    ) {
                    longest_name_id = i;
                    longest_length = bni;
                    shortest_length_from = from.size();
                  }
                  break;
                }

                if (nfromi == nfrom.size()) {
                  break;
                }

                if (std::tolower(bn[bni]) == std::tolower(nfrom[nfromi])) {
                  bni++;
                  nfromi++;
                  continue;
                }
                else if (bn[bni] == '_') {
                  bni++;
                }
                else {
                  break;
                }
              }
            }
          }
        }

        return longest_name_id;
      }
      std::string get_model_bone_name(uintptr_t name_index, auto& model) {
        std::string longest_name;
        uintptr_t longest_length = 0;
        uintptr_t shortest_length_from = (uintptr_t)-1;

        model.iterate_bones(*model.root_bone, [&](fan_3d::model::bone_t& bone) {
          std::string from = bone.name;

          for (uintptr_t bni1 = 0; bni1 < bone_names_model[name_index].size(); bni1++) {
            std::string bn = bone_names_model[name_index][bni1];
            for (uintptr_t ip = 0; ip < from.size(); ip++) {
              uintptr_t bni = 0;
              auto nfrom = from.substr(ip);
              uintptr_t nfromi = 0;
              while (1) {
                if (bni == bn.size()) {
                  if (
                    (bni > longest_length) ||
                    (from.size() < shortest_length_from)
                    ) {
                    longest_name = from;
                    longest_length = bni;
                    shortest_length_from = from.size();
                  }
                  break;
                }

                if (nfromi == nfrom.size()) {
                  break;
                }

                if (std::tolower(bn[bni]) == std::tolower(nfrom[nfromi])) {
                  bni++;
                  nfromi++;
                  continue;
                }
                else if (bn[bni] == '_') {
                  bni++;
                }
                else {
                  break;
                }
              }
            }
          }
          });

        return longest_name;
      }
      std::string get_bone_name_by_index(auto& model, uintptr_t index) {
        std::string ret;
        uintptr_t i = 0;
        model.iterate_bones(*model.root_bone, [&](fan_3d::model::bone_t& bone) {
          if (i == index) {
            ret = bone.name;
          }
          i++;
          });
        return ret;
      }
      void solve_legs(auto& iterator, auto& vector) {
        uint8_t left_leg_meaning = (uint8_t)-1;
        std::vector<std::vector<std::string>> left_leg_vector = {
          {"Left_leg"}
        };
        bool left_leg = false;
        iterator.iterate_bones(*iterator.root_bone, [&](fan_3d::model::bone_t& bone) {
          if (get_bone_name_index(left_leg_vector, bone.name) == 0) {
            if (left_leg) {
              /* multiple left leg? */
              assert(0);
            }
            left_leg = true;
          }
          });
        if (left_leg) {
          std::vector<std::vector<std::string>> left_down_leg_vector = {
            {"Lower_Leg_L", "Left_knee"}
          };
          iterator.iterate_bones(*iterator.root_bone, [&](fan_3d::model::bone_t& bone) {
            if (get_bone_name_index(left_down_leg_vector, bone.name) == 0) {
              if (left_leg_meaning != (uint8_t)-1) {
                /* multiple left leg meaning? */
                assert(0);
              }
              left_leg_meaning = 0;
            }
            });
          std::vector<std::vector<std::string>> left_up_leg_vector = {
            {"Left_Up_Leg"}
          };
          iterator.iterate_bones(*iterator.root_bone, [&](fan_3d::model::bone_t& bone) {
            if (get_bone_name_index(left_up_leg_vector, bone.name) == 0) {
              if (left_leg_meaning != (uint8_t)-1) {
                /* multiple left leg meaning? */
                assert(0);
              }
              left_leg_meaning = 1;
            }
            });
        }
        if (left_leg_meaning != (uint8_t)-1) {
          uintptr_t index;
          if (left_leg_meaning == 0) {
            index = get_bone_name_index(bone_names_default, "Left_Up_Leg");
          }
          else if (left_leg_meaning == 1) {
            index = get_bone_name_index(bone_names_default, "Lower_Leg_L");
          }
          else {
            index = (uintptr_t)-1;
          }
          if (index == (uintptr_t)-1) {
            /* internal error */
            assert(0);
          }

          vector[index].push_back("Left_leg");
        }

        uint8_t right_leg_meaning = (uint8_t)-1;
        std::vector<std::vector<std::string>> right_leg_vector = {
          {"Right_leg"}
        };
        bool right_leg = false;
        iterator.iterate_bones(*iterator.root_bone, [&](fan_3d::model::bone_t& bone) {
          if (get_bone_name_index(right_leg_vector, bone.name) == 0) {
            if (right_leg) {
              /* multiple right leg? */
              assert(0);
            }
            right_leg = true;
          }
          });
        if (right_leg) {
          std::vector<std::vector<std::string>> right_down_leg_vector = {
            {"Lower_Leg_R", "Right_knee"}
          };
          iterator.iterate_bones(*iterator.root_bone, [&](fan_3d::model::bone_t& bone) {
            if (get_bone_name_index(right_down_leg_vector, bone.name) == 0) {
              if (right_leg_meaning != (uint8_t)-1) {
                /* multiple right leg meaning? */
                assert(0);
              }
              right_leg_meaning = 0;
            }
            });
          std::vector<std::vector<std::string>> right_up_leg_vector = {
            {"Right_Up_Leg"}
          };
          iterator.iterate_bones(*iterator.root_bone, [&](fan_3d::model::bone_t& bone) {
            if (get_bone_name_index(right_up_leg_vector, bone.name) == 0) {
              if (right_leg_meaning != (uint8_t)-1) {
                /* multiple right leg meaning? */
                assert(0);
              }
              right_leg_meaning = 1;
            }
            });
        }
        if (right_leg_meaning != (uint8_t)-1) {
          uintptr_t index;
          if (right_leg_meaning == 0) {
            index = get_bone_name_index(bone_names_default, "Right_Up_Leg");
          }
          else if (right_leg_meaning == 1) {
            index = get_bone_name_index(bone_names_default, "Lower_Leg_R");
          }
          else {
            index = (uintptr_t)-1;
          }
          if (index == (uintptr_t)-1) {
            /* internal error */
            assert(0);
          }

          vector[index].push_back("Right_leg");
        }
      }
      // returns animation index from current scene
      uint32_t init_animation_import(std::string& animation_name, aiAnimation* newAnim) {
        bool animation_exists = false;
        for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
          if (scene->mAnimations[i]->mName.C_Str() == animation_name) {
            delete scene->mAnimations[i];
            scene->mAnimations[i] = newAnim;
            animation_exists = true;
            return i;
          }
        }

        if (!animation_exists) {
          aiAnimation** newAnimations = new aiAnimation * [scene->mNumAnimations + 1];
          for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
            newAnimations[i] = scene->mAnimations[i];
          }
          newAnimations[scene->mNumAnimations] = newAnim;
          newAnimations[scene->mNumAnimations]->mName = animation_name;
          if (scene->mAnimations) {
            delete[] scene->mAnimations;
          }
          scene->mAnimations = newAnimations;
          scene->mNumAnimations += 1;
          return scene->mNumAnimations - 1;
        }
        __unreachable();
        return -1;
      }
      std::unordered_map<std::string, bool> bone_mapper;
      static fan::mat4 get_world_matrix(fan_3d::model::bone_t* entity, fan::mat4 localMatrix) {
        fan_3d::model::bone_t* parentID = entity->parent;
        while (parentID != nullptr)
        {
          localMatrix = localMatrix * parentID->get_local_matrix();
          parentID = parentID->parent;
        }
        return localMatrix;
      }
      static fan::mat4 get_inverse_parent_matrix(fan_3d::model::bone_t* entity) {
        fan::mat4 inverseParentMatrix(1);
        fan_3d::model::bone_t* parentID = entity->parent;
        if (parentID != nullptr)
        {
          while (parentID != nullptr)
          {
            inverseParentMatrix = inverseParentMatrix * parentID->get_local_matrix();
            parentID = parentID->parent;
          }
          inverseParentMatrix = inverseParentMatrix.inverse();
        }
        return inverseParentMatrix;
      }
      static fan::vec3 transformPosition(
        const fan::vec3& position,
        bone_t& source_bone,
        bone_t& target_bone
      ) {
        fan_3d::model::bone_t transform = source_bone;
        transform.position = position;

        fan::mat4 local_matrix = source_bone.world_matrix.inverse() * fan_3d::model::fms_t::get_world_matrix(&source_bone, transform.get_local_matrix());
        local_matrix = target_bone.world_matrix * local_matrix * target_bone.inverse_parent_matrix;
        return target_bone.position;
      }
      static fan::quat transformRotation(
        fan::quat animation_rotation,
        bone_t& source_bone,
        bone_t& target_bone
      )
      {
        fan::quat tpose_adjust;
        fan::vec3 sangs, tangs;
        source_bone.rotation.to_angles(sangs);
        target_bone.rotation.to_angles(tangs);
        sangs.x = fan::math::degrees(sangs.x);
        sangs.y = fan::math::degrees(sangs.x);
        sangs.z = fan::math::degrees(sangs.z);

        tangs.x = fan::math::degrees(tangs.x);
        tangs.y = fan::math::degrees(tangs.y);
        tangs.z = fan::math::degrees(tangs.z);
        if (target_bone.name == "Left_arm") {
          tpose_adjust = fan::quat::from_angles(fan::vec3(-fan::math::radians(45), 0, 0));
        }
        if (target_bone.name == "Right_arm") {
          tpose_adjust = fan::quat::from_angles(fan::vec3(-fan::math::radians(45), 0, 0));
        }

        fan_3d::model::bone_t transform = source_bone;
        transform.rotation = (animation_rotation * tpose_adjust).normalize();

        fan::mat4 local_matrix = source_bone.world_matrix.inverse() * fan_3d::model::fms_t::get_world_matrix(&source_bone, transform.get_local_matrix());
        local_matrix = target_bone.world_matrix * local_matrix * target_bone.inverse_parent_matrix;
        return fan::quat(local_matrix).inverse();
      }
      static fan::vec3 transformScale(
        const fan::vec3& scale,
        bone_t& source_bone,
        bone_t& target_bone
      ) {
        fan_3d::model::bone_t transform = source_bone;
        transform.scale = scale;

        fan::mat4 local_matrix = source_bone.world_matrix.inverse() * fan_3d::model::fms_t::get_world_matrix(&source_bone, transform.get_local_matrix());
        local_matrix = target_bone.world_matrix * local_matrix * target_bone.inverse_parent_matrix;
        return local_matrix.get_scale();
      }
      void import_animation(fms_t& anim, const std::string& custom_name = "") {
        bone_names_anim = bone_names_default;
        bone_names_model = bone_names_default;
        if (anim.scene->mNumAnimations == 0) {
          fan::print("No animations in given animation.");
          return;
        }
        bone_mapper.clear();
        // only supports one animation import
        const aiAnimation* srcAnim = anim.scene->mAnimations[0];
        std::string animation_name = custom_name.empty() ? srcAnim->mName.C_Str() : custom_name;

        uint32_t animation_index = init_animation_import(animation_name, new aiAnimation(*srcAnim));
        aiAnimation* anim_ptr = scene->mAnimations[animation_index];
        solve_legs(anim, bone_names_anim);
        solve_legs(*this, bone_names_model);

        anim.iterate_bones(*anim.root_bone, [&](fan_3d::model::bone_t& bone) {
          auto bone_name_index = get_bone_name_index(bone_names_anim, bone.name);
          if (bone_name_index == (uintptr_t)-1) {
            printf("f \"%s\"\n", bone.name.c_str());
          }
          else {
            auto solved_bone_name = get_model_bone_name(bone_name_index, *this);
            if (solved_bone_name.size() == 0) {
              printf("nu \"%s\" -> \"%s\"\n", bone.name.c_str(), bone_names_default[bone_name_index][0].c_str());
            }
            else {
              printf("yay \"%s\" -> \"%s\" -> \"%s\"\n", bone.name.c_str(), bone_names_default[bone_name_index][0].c_str(), solved_bone_name.c_str());
              if (bone_mapper.find(solved_bone_name) != bone_mapper.end()) {
                fan::throw_error("duplicate bone from mapper:" + solved_bone_name, ", there must not be duplicate bones from mapper");
                return;
              }

              aiNodeAnim* channel = nullptr;
              for (unsigned int i = 0; i < anim_ptr->mNumChannels; ++i) {
                std::string check_str = anim_ptr->mChannels[i]->mNodeName.C_Str();
                if (check_str == bone.name) {
                  channel = anim_ptr->mChannels[i];
                  break;
                }
              }
              if (channel == nullptr) {
                return;
              }
              std::string original_name = channel->mNodeName.C_Str();

              // need to transform original bone from animation position, rotation, scale animation keys
              channel->mNodeName = bone_map[solved_bone_name]->name;

              fms_t& bone_source = anim;
              fms_t& bone_dest = *this;
              bone_t& source_bone = bone;
              auto found = bone_map.find(solved_bone_name);
              if (found == bone_map.end()) {
                fan::throw_error("invalid solved bone");
              }
              bone_t& target_bone = *bone_map[solved_bone_name];
              for (unsigned int i = 0; i < channel->mNumPositionKeys; i++) {
                fan::vec3 source_pos = channel->mPositionKeys[i].mValue;
                channel->mPositionKeys[i].mValue = transformPosition(source_pos, source_bone, target_bone);
              }

              for (unsigned int i = 0; i < channel->mNumRotationKeys; i++) {
                aiQuaternion sourceRotation = channel->mRotationKeys[i].mValue;
                channel->mRotationKeys[i].mValue = transformRotation(sourceRotation, source_bone, target_bone);
              }

              // For Scaling
              for (unsigned int i = 0; i < channel->mNumScalingKeys; i++) {
                fan::vec3 source = channel->mScalingKeys[i].mValue;
                //const aiVector3D position(source_bone.transform *  parentToBone * channel->mScalingKeys[i].mValue);

                //channel->mScalingKeys[i].mValue = *(fan::vec3*)&position;
                channel->mScalingKeys[i].mValue = transformScale(source, source_bone, target_bone);
              }


              bone_mapper[solved_bone_name];
            }
          }
        });
        animation_list.clear();
        load_animations();
      }

      f32_t longest_animation = 1.f;
      anim_key_t active_anim;
      std::unordered_map<anim_key_t, animation_data_t> animation_list;

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

        bool node_open = ImGui::TreeNodeEx((bone->name + " | parent " + (bone->parent ? bone->parent->name : "N/A")).c_str(), flags);

        if (node_open) {
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
              bone->rotation = fan::quat();
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
                auto& bt = anim.bone_transform_tracks[bone.id];
                uint32_t data_count = bt.rotation_timestamps.size();
                if (data_count) {
                  bool node_open = ImGui::TreeNode(bone.name.c_str());
                  if (node_open) {
                    for (int i = 0; i < bt.rotation_timestamps.size(); ++i) {
                      ImGui::DragFloat(
                        ("rotation:" + std::to_string(i)).c_str(),
                        &bt.rotation_timestamps[i]
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
                auto& bt = anim.bone_transform_tracks[bone.id];
                uint32_t data_count = bt.rotations.size();
                if (data_count) {
                  bool node_open = ImGui::TreeNode(bone.name.c_str());
                  if (node_open) {
                    for (int i = 0; i < bt.rotations.size(); ++i) {
                      ImGui::DragFloat4(
                        ("rotation:" + std::to_string(i)).c_str(),
                        bt.rotations[i].data()
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

      void mouse_modify_joint(f32_t ddt) {
        static constexpr f64_t delta_time_divier = 1e+6;
        ImGui::DragFloat("current time", &dt, 1, 0, std::max(longest_animation, 1.f));
        //ImGui::Text(fan::format("camera pos: {}\ntotal time: {:.2f}", gloco->default_camera_3d->camera.position, fms.default_animation.duration).c_str());

        if (ImGui::Checkbox("play animation", &play_animation)) {
          showing_temp_rot = false;
        }
        if (play_animation) {
          dt += ddt * 1000;
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
          auto& bt = anim.bone_transform_tracks[active_bone];
          for (int i = 0; i < bt.rotations.size(); ++i) {
            ImGui::DragFloat4(("rotations:" + std::to_string(i)).c_str(), bt.rotations[i].data(), 0.01);
          }

          static int32_t current_frame = 0;
          if (!play_animation) {
            dt = current_frame;
          }
          else {
            current_frame = fmodf(dt, anim.duration);
          }

          static f32_t prev_frame = 0;
          if (prev_frame != dt) {
            showing_temp_rot = false;
            prev_frame = dt;
          }

          int32_t start_frame = 0;
          int32_t end_frame = std::ceil(anim.duration);
          if (end_frame <= start_frame) {
            ImGui::Text("No bones in current animation");
            return;
          }

          if (ImGui::Button("save keyframe")) {
            fk_set_rot(active_anim, active_bone, current_frame / 1000.f, anim.bone_poses[active_bone].rotation);
          }
        }
      }

      int active_bone = -1;
      bool toggle_rotate = false;
      bool showing_temp_rot = false;
      bool play_animation = false;

      // ---------------------gui---------------------

      aiScene* scene = nullptr;
      bone_t* root_bone = nullptr;
      Assimp::Importer importer;

      std::vector<pm_material_data_t> material_data_vector;
      std::vector<fan::mat4> bone_transforms;
      std::vector<mesh_t> meshes;
      std::vector<mesh_t> calculated_meshes;

      struct ci_less {
        // case-independent (ci) compare_less binary function
        struct nocase_compare
        {
          bool operator() (const unsigned char& c1, const unsigned char& c2) const {
            return tolower(c1) < tolower(c2);
          }
        };
        bool operator() (const std::string& s1, const std::string& s2) const {
          return std::lexicographical_compare
          (s1.begin(), s1.end(),   // source range
            s2.begin(), s2.end(),   // dest range
            nocase_compare());  // comparison
        }
      };
      std::map<std::string, bone_t*, ci_less> bone_map;
      std::vector<bone_t*> bones;
      properties_t p;
      fan::mat4 m_transform{ 1 };
      uint32_t bone_count = 0;
      f32_t dt = 0;
      f32_t scale_divider = 1;
    };
  }
}

#ifdef fms_use_opengl
  #undef fms_use_opengl
#endif