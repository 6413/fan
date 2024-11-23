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
      fan::vec3 vertex1;
      fan::vec3 tangent;
      fan::vec3 bitangent;
    };

    // structure to hold bone tree (skeleton)
    struct joint_t {
      int id;
      std::string name;
      fan::mat4 offset;
      fan::mat4 global_transform;
      fan::mat4 local_transform;
      std::vector<joint_t> children;
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

    struct Bone {
      std::string name;
      fan::mat4 offset;
      fan::mat4 transform;
      std::vector<Bone*> children;
      Bone* parent;

      float rotationX = 0.0f;
      float rotationY = 0.0f;
      float rotationZ = 0.0f;
    };

    struct mesh_t {
      std::vector<fan_3d::model::vertex_t> vertices;
      std::vector<unsigned int> indices;

      fan::opengl::core::vao_t VAO;
      fan::opengl::core::vbo_t VBO;
      unsigned int EBO;
    };

    // pm -- parsed model


    struct pm_texture_data_t {
      fan::vec2ui size = 0;
      std::vector<uint8_t> data;
      int channels = 0;
    };
    struct pm_material_data_t {
      std::string texture_id[AI_TEXTURE_TYPE_MAX + 1];
      fan::vec4 color[AI_TEXTURE_TYPE_MAX + 1];
    };
    inline static std::unordered_map<std::string, pm_texture_data_t> cached_texture_data;
    inline static std::vector<pm_material_data_t> material_data_vector;

    struct pm_model_data_t {
      struct mesh_data_t {
        // names is generated for each texture type to cache the texture data
        std::array<std::string, AI_TEXTURE_TYPE_MAX + 1> names;
      };
      // mesh[]
      std::vector<std::vector<fan_3d::model::vertex_t>> vertices;
      std::vector<mesh_data_t> mesh_data;
      fan_3d::model::joint_t skeleton;
      uint32_t bone_count = 0;
    };


    struct parsed_model_t {
      pm_model_data_t model_data;
      std::vector<fan::mat4> transforms;
    };

    inline std::vector<int> mesh_id_table;

    static std::string load_texture(
      const aiScene* scene,
      aiMaterial* material,
      aiTextureType texture_type,
      const fan::string& root_path,
      parsed_model_t& parsed_model,
      std::size_t mesh_index
    ){
      aiString path;
      if (material->GetTexture(texture_type, 0, &path) != AI_SUCCESS) {
        return std::string("");
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
        parsed_model.model_data.mesh_data[mesh_index].names[texture_type] = generated_str;
        auto& td = cached_texture_data[generated_str];
        td.size = fan::vec2(width, height);
        td.data.insert(td.data.end(), data, data + td.size.multiply() * nr_channels);
        td.channels = nr_channels;
        stbi_image_free(data);
        return std::string(path.C_Str());
      }
      else {
        fan::throw_error("help");
        return std::string("");
        #if 0
        fan::string file_path = root_path + "textures/" + scene->GetShortFilename(path.C_Str());

        parsed_model.model_data.mesh_data[mesh_index].names[texture_type] = file_path;
        auto found = cached_texture_data.find(file_path);
        if (found == cached_texture_data.end()) {
          fan::print(file_path);
          texture_found = true;

          fan::image::image_info_t ii;
          fan::image::load(file_path, &ii);
          auto& td = cached_texture_data[file_path];
          td.size = ii.size;
          td.data.insert(td.data.end(), (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * ii.channels);
          td.channels = ii.channels;
          fan::image::free(&ii);
        }
        #endif
      }
    }

    static void load_animation(const aiScene* scene, fan_3d::model::animation_data_t& animation) {
      if (scene->mNumAnimations == 0) {
        return;
      }
      //loading  first Animation
      aiAnimation* anim = scene->mAnimations[0];

      animation.duration = anim->mDuration/* * anim->mTicksPerSecond*/;

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

    static bool get_time_fraction(const std::vector<f32_t>& times, f32_t dt, std::pair<uint32_t, f32_t>& fp) {
      if (times.empty()) {
        return true;
      }
      auto it = std::upper_bound(times.begin(), times.end(), dt);
      uint32_t segment = std::distance(times.begin(), it);
      if (times.size() == 1) {
        fp = { 0,  std::clamp((dt) / (times[0]), 0.0f, 1.0f) };
        return false;
      }
      if (segment == 0) {
        segment++;
      }

      segment = fan::clamp(segment, uint32_t(0), uint32_t(times.size() - 1));

      f32_t start = times[segment - 1];
      f32_t end = times[segment];
      fp = { segment, std::clamp((dt - start) / (end - start), 0.0f, 1.0f) };
      return false;
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
    struct fms_model_info_t {
      fan::string path;
    };

    // fan model stuff
    struct fms_t {

      const aiScene* scene;

      fms_t() = default;
      fms_t(const fms_model_info_t& fmi) {
        if (!load_model(fmi.path)) {
          fan::throw_error("failed to load model:" + fmi.path);
        }
        this->scene = scene;
        calculated_meshes = meshes;
      }

      void process_skeleton(aiNode* node, Bone* parent) {
        Bone* bone = new Bone();
        bone->name = node->mName.C_Str();
        bone->parent = parent;

        // Convert aiMatrix4x4 to glm::mat4
        aiMatrix4x4 transform = node->mTransformation;
        bone->transform = transform;

        // Initialize offset matrix (will be updated if this is a bone in the mesh)
        bone->offset = fan::mat4(1.0f);

        // Update bone mapping
        boneMap[bone->name] = bone;

        // Set as root if no parent
        if (parent == nullptr) {
          rootBone = bone;
        }
        else {
          parent->children.push_back(bone);
        }

        // Process all child nodes recursively
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
          process_skeleton(node->mChildren[i], bone);
        }
      }

      // kinda illegal here
      void SetupMeshBuffers(fan_3d::model::mesh_t& mesh) {
        mesh.VAO.open(*gloco);
        mesh.VBO.open(*gloco, GL_ARRAY_BUFFER);
        gloco->opengl.glGenBuffers(1, &mesh.EBO);

        mesh.VAO.bind(*gloco);

        mesh.VBO.bind(*gloco);
        gloco->opengl.glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(fan_3d::model::vertex_t), &mesh.vertices[0], GL_STATIC_DRAW);

        gloco->opengl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
        gloco->opengl.glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW);


        gloco->get_context().opengl.glEnableVertexAttribArray(0);
        gloco->get_context().opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, position));
        gloco->get_context().opengl.glEnableVertexAttribArray(1);
        gloco->get_context().opengl.glVertexAttribPointer(1, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, normal));
        gloco->get_context().opengl.glEnableVertexAttribArray(2);
        gloco->get_context().opengl.glVertexAttribPointer(2, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, uv));
        gloco->get_context().opengl.glEnableVertexAttribArray(3);
        gloco->get_context().opengl.glVertexAttribPointer(3, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bone_ids));
        gloco->get_context().opengl.glEnableVertexAttribArray(4);
        gloco->get_context().opengl.glVertexAttribPointer(4, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bone_weights));
        gloco->get_context().opengl.glEnableVertexAttribArray(5);
        gloco->get_context().opengl.glVertexAttribPointer(5, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, vertex1));
        gloco->get_context().opengl.glEnableVertexAttribArray(6);
        gloco->get_context().opengl.glVertexAttribPointer(6, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, tangent));
        gloco->get_context().opengl.glEnableVertexAttribArray(7);
        gloco->get_context().opengl.glVertexAttribPointer(7, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, bitangent));
        gloco->get_context().opengl.glBindVertexArray(0);
      }

      void ProcessBoneOffsets(aiMesh* mesh) {
        for (unsigned int i = 0; i < mesh->mNumBones; i++) {
          aiBone* bone = mesh->mBones[i];
          std::string boneName = bone->mName.C_Str();

          // Find the bone in our map
          auto it = boneMap.find(boneName);
          if (it != boneMap.end()) {
            // Convert offset matrix
            aiMatrix4x4 offset = bone->mOffsetMatrix;
            it->second->offset = offset;
          }
        }
      }

      mesh_t ProcessMesh(aiMesh* mesh) {
        mesh_t newMesh;
        std::vector<vertex_t> tempVertices(mesh->mNumVertices);

        // Initialize vertices with zero weights and invalid bone IDs
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
          vertex_t& vertex = tempVertices[i];

          // Position
          vertex.position = fan::vec3(
            mesh->mVertices[i].x,
            mesh->mVertices[i].y,
            mesh->mVertices[i].z
          );

          // Normal
          if (mesh->HasNormals()) {
            vertex.normal = fan::vec3(
              mesh->mNormals[i].x,
              mesh->mNormals[i].y,
              mesh->mNormals[i].z
            );
          }

          // TexCoords
          if (mesh->mTextureCoords[0]) {
            vertex.uv = fan::vec2(
              mesh->mTextureCoords[0][i].x,
              mesh->mTextureCoords[0][i].y
            );
          }
          else {
            vertex.uv = fan::vec2(0.0f);
          }

          // Initialize bone data
          vertex.bone_ids = fan::vec4i(-1);  // -1 indicates no bone assigned
          vertex.bone_weights = fan::vec4(0.0f);
        }

        // Process bone weights
        for (uint32_t i = 0; i < mesh->mNumBones; i++) {
          aiBone* bone = mesh->mBones[i];
          for (uint32_t j = 0; j < bone->mNumWeights; j++) {
            uint32_t vertexId = bone->mWeights[j].mVertexId;
            float weight = bone->mWeights[j].mWeight;

            // Find first available slot in vertex bone data
            for (int k = 0; k < 4; k++) {
              if (tempVertices[vertexId].bone_weights[k] == 0.0f) {
                tempVertices[vertexId].bone_ids[k] = i;
                tempVertices[vertexId].bone_weights[k] = weight;
                break;
              }
            }
          }
        }

        // Normalize weights and handle unassigned vertices
        for (auto& vertex : tempVertices) {
          float weightSum = vertex.bone_weights.x + vertex.bone_weights.y +
            vertex.bone_weights.z + vertex.bone_weights.w;

          if (weightSum > 0.0f) {
            // Normalize existing weights
            vertex.bone_weights /= weightSum;
          }
          else {
            // If no bones influence this vertex, assign it fully to the first bone
            vertex.bone_ids.x = 0;
            vertex.bone_weights.x = 1.0f;
          }

          // Ensure no invalid bone IDs
          for (int i = 0; i < 4; i++) {
            if (vertex.bone_ids[i] < 0) {
              vertex.bone_ids[i] = 0;
              vertex.bone_weights[i] = 0.0f;
            }
          }
        }

        newMesh.vertices = tempVertices;

        // Process indices
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
          aiFace& face = mesh->mFaces[i];
          for (unsigned int j = 0; j < face.mNumIndices; j++) {
            newMesh.indices.push_back(face.mIndices[j]);
          }
        }

        // Debug output
        //std::cout << "Processed mesh with " << mesh->mNumVertices << " vertices and " 
       ///           << mesh->mNumBones << " bones" << std::endl;

        SetupMeshBuffers(newMesh);
        return newMesh;
      }

      void UpdateBoneRotation(const std::string& boneName, float x, float y, float z) {
        auto it = boneMap.find(boneName);
        if (it != boneMap.end()) {
          Bone* bone = it->second;
          bone->rotationX = x;
          bone->rotationY = y;
          bone->rotationZ = z;
        }
      }
      void UpdateBoneTransforms(Bone* bone, const fan::mat4& parentTransform, std::vector<fan::mat4>& transforms) {
          if (!bone) return;

          fan::mat4 rotation = fan::mat4(1.0f);

          fan::mat4 localRotation = fan::mat4(1.0f);
          localRotation = localRotation.rotate(fan::math::radians(bone->rotationX), fan::vec3(1, 0, 0));
          localRotation = localRotation.rotate(fan::math::radians(bone->rotationY), fan::vec3(0, 1, 0));
          localRotation = localRotation.rotate(fan::math::radians(bone->rotationZ), fan::vec3(0, 0, 1));

          fan::mat4 localTransform = localRotation * bone->transform;

          fan::mat4 globalTransform = parentTransform * localTransform;

          transforms.push_back(globalTransform * bone->offset);

          for (Bone* child : bone->children) {
            UpdateBoneTransforms(child, globalTransform, transforms);
          }
        }

      void UpdateAllBones(std::vector<fan::mat4>& transforms) {
        transforms.clear();
        if (rootBone) {
          UpdateBoneTransforms(rootBone, m_transform, transforms);
        }
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

        for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
          mesh_t newMesh = ProcessMesh(scene->mMeshes[i]);
          ProcessBoneOffsets(scene->mMeshes[i]);
          meshes.push_back(newMesh);
        }

        UpdateAllBones(bone_transforms);

        // Setup bone buffer
        // IF GPU
        /*GLuint boneTransformSize = boneTransforms.size() * sizeof(glm::mat4);
        engine.opengl.glGenBuffers(1, &boneBuffer);
        engine.opengl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, boneBuffer);
        engine.opengl.glBufferData(GL_SHADER_STORAGE_BUFFER, boneTransformSize, &boneTransforms[0], GL_STATIC_DRAW);
        engine.opengl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boneBuffer);*/
        return true;
      }


      fan::vec4 calculate_bone_transform(uint32_t mesh_id, uint32_t vertex_id, const std::vector<fan::mat4>& bone_transforms) {
        auto& vertex = meshes[mesh_id].vertices[vertex_id];
        fan::vec4i bone_ids = vertex.bone_ids;
        fan::vec4 bone_weights =vertex.bone_weights;

        fan::vec4 totalPosition(0.0);
        float weightSum = 0.0;

        for (int i = 0; i < 4; i++) {
          float weight = vertex.bone_weights[i];
          int boneId = vertex.bone_ids[i];

          if (weight > 0.0 && boneId >= 0) {
            weightSum += weight;
            fan::mat4 transform = bone_transforms[boneId];
            totalPosition += transform * fan::vec4(vertex.position, 1.0) * weight;
          }
        }
        if(weightSum < 0.001) {
            totalPosition = fan::vec4(vertex.position, 1.0);
        }

        fan::mat4 model(1);
        fan::vec4 worldPos = model * totalPosition;
        fan::vec4 gl_Position = worldPos;
        return gl_Position;
      }

      std::vector<fan::mat4> fk_calculate_transformations() {
        UpdateAllBones(bone_transforms);
        return bone_transforms;
      }

      void calculate_modified_vertices(uint32_t mesh_id, const std::vector<fan::mat4>& transformations) {
        calculate_modified_vertices(mesh_id, fan::mat4(1), transformations);
      }

      // converts right hand coordinate to left hand coordinate
      void calculate_modified_vertices(uint32_t mesh_id, const fan::mat4& model, const std::vector<fan::mat4>& transformations) {

        for (int i = 0; i < meshes[mesh_id].vertices.size(); ++i) {

          fan::vec3 v = meshes[mesh_id].vertices[i].position;
          if (transformations.empty()) {

            fan::vec4 vertex_position = m_transform * model * fan::vec4(
              v
              , 1.0);
            calculated_meshes[mesh_id].vertices[i].position = fan::vec3(vertex_position.x, vertex_position.y, vertex_position.z);
          }
          else {
            fan::vec4 interpolated_bone_transform = calculate_bone_transform(mesh_id, i, transformations);

            fan::vec4 vertex_position = fan::vec4(v, 1.0);

            fan::vec4 result = interpolated_bone_transform;

            calculated_meshes[mesh_id].vertices[i].position = fan::vec3(result.x, result.y, result.z);
          }

        }
      }

      // parses joint data @ dt
      bool fk_parse_joint_data(const bone_transform_track_t& btt, fan::vec3& position, fan::quat& rotation, fan::vec3& scale) {
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

      void fk_get_pose(animation_data_t& animation, joint_t& joint, fan::mat4& parent_transform) {

      }

      void fk_interpolate_animations(std::vector<fan::mat4>& joint_transforms, animation_data_t& animation, joint_t& joint, fan::mat4& parent_transform, f32_t animation_weight) {

        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;

        fan::mat4 global_transform;

        if (animation_list.empty()) {
          // tpose - DEAD
          fan::throw_error("DEAD");
          joint_transforms[joint.id] = joint.global_transform * joint.offset;
        }
        else {
          for (auto& apair : animation_list) {
            auto& a = apair.second;
            position += a.joint_poses[joint.id].position * a.weight;
            fan::quat slerped_quat = fan::quat::slerp(fan::quat(1, 0, 0, 0), a.joint_poses[joint.id].rotation, a.weight);
            rotation = (rotation * slerped_quat).normalize();
            scale += a.joint_poses[joint.id].scale * a.weight;
          }
          //
          fan::mat4 local_transform = joint.local_transform;
                    //fan::mat4 rot = fan::mat4(1).rotate(fan::math::radians(90.f));
          //local_transform = local_transform.translate(position);
         // local_transform = local_transform.rotate(fan::math::radians(45.f), fan::vec3(1, 0, 0));
          //local_transform = local_transform.scale(scale);
          //animation.pose[joint.id]
          global_transform = parent_transform* joint.offset;

          //animation.pose[joint.id] = global_transform * joint.offset;
          joint_transforms[joint.id] = global_transform * joint.local_transform;
         // global_transform = parent_transform * local_transform;
        }

        for (fan_3d::model::joint_t& child : joint.children) {
          fk_interpolate_animations(joint_transforms, animation, child, global_transform, animation_weight);
        }
      }

      struct one_triangle_t {
        fan::vec3 p[3];
        fan::vec2 tc[3];
      };

      void get_triangle_vec(uint32_t mesh_id, std::vector<one_triangle_t>* triangles) {
        static constexpr int edge_count = 3;
        // ignore i for now since only one scene
        triangles->resize(calculated_meshes[mesh_id].vertices.size() / edge_count);
        for (int i = 0; i < calculated_meshes[mesh_id].vertices.size(); ++i) {
          (*triangles)[i / edge_count].p[i % edge_count] = calculated_meshes[mesh_id].vertices[i].position;
          (*triangles)[i / edge_count].tc[i % edge_count] = calculated_meshes[mesh_id].vertices[i].uv;
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

      f32_t dt = 0;

      fan::mat4 m_transform{1};
      std::vector<fan::mat4> bone_transforms;

      std::vector<mesh_t> meshes;
      // for bone transformations
      std::vector<mesh_t> calculated_meshes;
      std::map<std::string, Bone*> boneMap;
      Bone* rootBone = nullptr;
    };
  }
}