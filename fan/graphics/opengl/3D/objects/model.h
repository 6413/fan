namespace fan_3d {
  namespace model {

    struct vertex_t {
      fan::vec3 position;
      fan::vec3 normal;
      fan::vec2 uv;
      fan::vec4 bone_ids;
      fan::vec4 bone_weights;
      fan::vec3 vertex1;
      fan::vec3 tangent;
      fan::vec3 bitangent;
      fan::color diffuse;
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
      f_t duration;
      // ticks per second
      f_t tps;
      std::unordered_map<std::string, bone_transform_track_t> bone_transforms;
      struct joint_pose_t {
        fan::vec3 position;
        fan::quat rotation;
        fan::vec3 scale;
      };
      std::vector<joint_pose_t> joint_poses;
      std::vector<fan::mat4> pose;
      f32_t weight;
    };

    // a recursive function to read all bones and form skeleton
    // std::unordered_map<std::string, std::pair<int, fan::mat4>> bone_info_table
    bool read_skeleton(auto This, fan_3d::model::joint_t& joint, aiNode* node, auto& bone_info_table, const fan::mat4& parent_transform) {

      if (bone_info_table.find(node->mName.C_Str()) != bone_info_table.end()) { // if node is actually a bone

        aiMatrix4x4 global_transform = parent_transform * node->mTransformation;

        joint.name = node->mName.C_Str();
        joint.id = bone_info_table[joint.name].first;
        joint.offset = bone_info_table[joint.name].second;

        joint.local_transform = node->mTransformation;
        joint.global_transform = global_transform;

        This->bone_strings.push_back(joint.name);

        for (int i = 0; i < node->mNumChildren; i++) {
          fan_3d::model::joint_t child;
          read_skeleton(This, child, node->mChildren[i], bone_info_table, global_transform);
          joint.children.push_back(child);
        }
        return true;
      }
      else { // find bones in children
        for (int i = 0; i < node->mNumChildren; i++) {
          if (read_skeleton(This, joint, node->mChildren[i], bone_info_table, parent_transform)) {
            return true;
          }

        }
      }
      return false;
    }

    // pm -- parsed model

    struct pm_texture_data_t {
      fan::vec2ui diffuse_texture_size;
      std::vector<uint8_t> diffuse_texture_data;
      std::vector<uint8_t> normal_texture_data;
      std::vector<uint8_t> roughness_texture_data;
      std::vector<uint8_t> metallic_texture_data;
    };

    struct pm_model_data_t {
      // mesh[]
      std::vector<std::vector<fan_3d::model::vertex_t>> vertices;
      fan_3d::model::joint_t skeleton;
      uint32_t bone_count = 0;
    };

    struct parsed_model_t {
      pm_model_data_t model_data;
      std::vector<fan::mat4> transforms;
      struct textures_t {
        std::string diffuse;
        std::string normal;
        std::string roughness;
        std::string metallic;
      };
      std::vector<textures_t> texture_names;
    };

    inline std::unordered_map<std::string, pm_texture_data_t> cached_texture_data;
    inline std::vector<int> mesh_id_table;

    void process_model(auto This, const fan::string& root_path, const aiScene* scene, aiNode* node, parsed_model_t& parsed_model) {
      std::cout << "Processing node: " << node->mName.C_Str() << std::endl;

      // Process all the node's meshes (if any)
      for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        load_model(This, root_path, scene, mesh, node, parsed_model);
        mesh_id_table.push_back(node->mNumMeshes);
      }

      // Then do the same for each of its children
      for (unsigned int i = 0; i < node->mNumChildren; i++) {
        process_model(This, root_path, scene, node->mChildren[i], parsed_model);
      }
    }

    // gets diffuse for now
    static bool load_material(aiMaterial* material, fan::color& o_diffuse) {
      aiColor4D diffuse;
      if (AI_SUCCESS == aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &diffuse)) {
        o_diffuse = fan::color(diffuse.r, diffuse.g, diffuse.b, diffuse.a);
        return false;
      }
      return true;
    }

    static bool load_texture(aiMaterial* material, aiTextureType texture_type, const fan::string& root_path, int channels_rgba, parsed_model_t& parsed_model, uint32_t& texture_offset)
    {
      bool texture_found = false;
      aiString path;
      if (material->GetTexture(texture_type, 0, &path) == AI_SUCCESS)
      {
        fan::string str = path.C_Str();
        auto idx = str.find_last_of('\\') + 1;
        fan::webp::image_info_t ii;
        str = root_path + "textures/" + str.substr(idx);
        fan::print(str);
        str.replace_all(".png", ".webp");

        auto found = cached_texture_data.find(str);
        if (found == cached_texture_data.end())
        {
          texture_found = true;
          if (fan::webp::load(/*root_path + path.C_Str()*/str, &ii))
          {
            fan::throw_error("failed to load image data from path:" + root_path + path.C_Str());
          }

          auto& d = cached_texture_data[str];
          d.diffuse_texture_size = ii.size;

          if (texture_type == aiTextureType_METALNESS)
          {
            d.metallic_texture_data.insert(
              d.metallic_texture_data.end(),
              (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * channels_rgba
            );
            parsed_model.texture_names[texture_offset].metallic = str;
          }
          else if (texture_type == aiTextureType_SHININESS)
          {
            d.roughness_texture_data.insert(
              d.roughness_texture_data.end(),
              (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * channels_rgba
            );
            parsed_model.texture_names[texture_offset].roughness = str;
          }
          else if (texture_type == aiTextureType_DIFFUSE)
          {
            d.diffuse_texture_data.insert(
              d.diffuse_texture_data.end(),
              (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * channels_rgba
            );
            parsed_model.texture_names[texture_offset].diffuse = str;
            {
              str.replace_all("BaseColor", "Normal");
              if (fan::io::file::exists(str)) {
                auto found = cached_texture_data.find(str);
                if (found == cached_texture_data.end()) {
                  if (fan::webp::load(/*root_path + path.C_Str()*/str, &ii)) {
                    fan::throw_error("failed to load image data from path:" + root_path + path.C_Str());
                  }

                  static constexpr int channels_rgba = 4;
                  auto& d = cached_texture_data[str];
                  d.normal_texture_data.insert(
                    d.normal_texture_data.end(),
                    (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * channels_rgba
                  );
                }
                parsed_model.texture_names[texture_offset].normal = str;
              }
            }
          }
        }
        else {
          switch (texture_type) {
            case aiTextureType_DIFFUSE: {
              parsed_model.texture_names[texture_offset].diffuse = str;
              str.replace_all("BaseColor", "Normal");
              parsed_model.texture_names[texture_offset].normal = str;
              break;
            }
            case aiTextureType_METALNESS: {
              parsed_model.texture_names[texture_offset].metallic = str;
              break;
            }
            case aiTextureType_SHININESS: {
              parsed_model.texture_names[texture_offset].roughness = str;
              break;
            }
          }
        }
      }
      return texture_found;
    }
    // converts indices to triangles only - todo use indices (needs bcol update)
    void load_model(auto This, const fan::string& root_path, const aiScene* scene, aiMesh* mesh, aiNode* node, parsed_model_t& parsed_model) {
      std::vector<fan_3d::model::vertex_t> temp_vertices;

      uint32_t vertex_offset = parsed_model.model_data.vertices.size();
      parsed_model.model_data.vertices.resize(parsed_model.model_data.vertices.size() + 1);
      auto& vertices = parsed_model.model_data.vertices[vertex_offset];
      uint32_t texture_offset = parsed_model.texture_names.size();
      parsed_model.texture_names.resize(parsed_model.texture_names.size() + 1);


      static constexpr int channels_rgba = 4;
      auto arr = std::to_array({
           aiTextureType_DIFFUSE, aiTextureType_SHININESS,
           aiTextureType_METALNESS
        });
      fan::color color_diffuse;
      for (auto& i : arr) {
        if (load_texture(scene->mMaterials[mesh->mMaterialIndex],
          i, root_path, channels_rgba, parsed_model, texture_offset)) {
        }
      }
      if (load_material(scene->mMaterials[mesh->mMaterialIndex], color_diffuse)) {
        fan::throw_error("failed to find material");
      }

      parsed_model.transforms.push_back(node->mTransformation);

      //load position, normal, uv
      for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
        //process position 
        fan_3d::model::vertex_t vertex;
        fan::vec3 vector;
        vector.x = mesh->mVertices[i].x;
        vector.y = mesh->mVertices[i].y;
        vector.z = mesh->mVertices[i].z;
        vertex.position = vector;
        //process normal
        if (mesh->mNormals) {
          vector.x = mesh->mNormals[i].x;
          vector.y = mesh->mNormals[i].y;
          vector.z = mesh->mNormals[i].z;
          vertex.normal = vector;
        }
        else {
          vertex.normal = 0;
        }
        //process uv
        if (mesh->mTextureCoords) {
          fan::vec2 vec;

          if (mesh->mTextureCoords[0]) {
            vec.x = mesh->mTextureCoords[0][i].x;
            vec.y = mesh->mTextureCoords[0][i].y;
            vertex.uv = vec;
          }
        }
        else {
          vertex.uv = 0;
        }
        if (mesh->mTangents) {
          vertex.vertex1 = mesh->mTangents[i];
        }
        if (mesh->mBitangents) {
          vertex.bitangent = mesh->mBitangents[i];
        }

        vertex.bone_ids = fan::vec4i(0);
        vertex.bone_weights = fan::vec4(0.0f);
        vertex.diffuse = color_diffuse;

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

      //load indices
      for (int i = 0; i < mesh->mNumFaces; i++) {
        aiFace& face = mesh->mFaces[i];
        for (uint32_t j = 0; j < face.mNumIndices; j++) {
          vertices.push_back(temp_vertices[face.mIndices[j]]);
          std::size_t idx = vertices.size() - 1;
          fan::vec4 vp = fan::mat4(node->mTransformation) * fan::vec4(vertices[idx].position, 1.0);
          fan::vec4 np = fan::mat4(node->mTransformation) * fan::vec4(vertices[idx].vertex1, 1.0);
          fan::vec3 right = *(fan::vec3*)&vp;
          // convert from right to left handed
          vertices[idx].position = fan::vec3(vp.x, vp.z, vp.y);
          vertices[idx].normal = *(fan::vec3*)&np;
        }
      }

      fan::mat4 global_transform(1);
      // create bone hirerchy
      read_skeleton(This, parsed_model.model_data.skeleton, scene->mRootNode, bone_info, global_transform);
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

    bool get_time_fraction(const std::vector<f32_t>& times, f32_t dt, std::pair<uint32_t, f32_t>& fp) {
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

    bool fk_get_time_fraction(const std::vector<f32_t>& times, f32_t dt, std::pair<uint32_t, f32_t>& fp) {
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

    void apply_transform(fan_3d::model::joint_t& bone, fan::mat4& rot_transform) {
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

        const aiScene* scene = importer.ReadFile(path,  aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
          fan::print("animation load error:", importer.GetErrorString());
          exit(1);
        }

        m_transform = scene->mRootNode->mTransformation;

        // aiMatrix4x4 globalTransform = getGlobalTransform(scene->mRootNode);

        std::string root_path = "";

        size_t last_slash_pos = path.find_last_of("/");
        if (last_slash_pos != std::string::npos) {
          root_path = path.substr(0, last_slash_pos + 1);
        }

        process_model(this, root_path, scene, scene->mRootNode, parsed_model);

        load_animation(scene, default_animation);

        tpose.joint_poses.resize(parsed_model.model_data.bone_count, { {}, {}, 1 });
        tpose.pose.resize(parsed_model.model_data.bone_count, fan::mat4(1)); // use this for no animation
        default_animation.pose.resize(parsed_model.model_data.bone_count, fan::mat4(1)); // use this for no animation
        temp_rotations.resize(parsed_model.model_data.bone_count);

        m_modified_verticies = parsed_model.model_data.vertices;
      }

      fan::mat4 calculate_bone_transform(uint32_t vertex_id, const std::vector<fan::mat4>& bone_transforms) {
        static constexpr uint32_t mesh_id = 0;
        fan::vec4i bone_ids = parsed_model.model_data.vertices[mesh_id][vertex_id].bone_ids;
        fan::vec4 bone_weights = parsed_model.model_data.vertices[mesh_id][vertex_id].bone_weights;

        fan::mat4 bone_transform = fan::mat4(0);
        bone_transform += bone_transforms[bone_ids.x] * bone_weights.x;
        bone_transform += bone_transforms[bone_ids.y] * bone_weights.y;
        bone_transform += bone_transforms[bone_ids.z] * bone_weights.z;
        bone_transform += bone_transforms[bone_ids.w] * bone_weights.w;
        return bone_transform;
      }

      // for default animation
      std::vector<fan::mat4> calculate_transformations() {
        std::vector<fan::mat4> transformations;
        transformations.resize(default_animation.pose.size(), fan::mat4(1));
        fan::mat4 initial(1);
        initial = initial.rotate(-fan::math::pi / 2, fan::vec3(1, 0, 0));
        interpolate_default_animation(transformations);
        return transformations;
      }

      std::vector<fan::mat4> fk_calculate_transformations() {
        fk_transformations.resize(default_animation.pose.size(), fan::mat4(1));
        fan::mat4 initial(1);
        initial = initial.rotate(-fan::math::pi / 2, fan::vec3(1, 0, 0));

        static f32_t animation_weight = 1;
        ImGui::DragFloat("weight", &animation_weight, 0.1, 0, 1.0);

        fk_interpolate_animations(fk_transformations, default_animation, parsed_model.model_data.skeleton, initial, animation_weight);
        return fk_transformations;
      }
      f32_t x = 0;

      void calculate_modified_vertices(uint32_t mesh_id, const fan::mat4& model, const std::vector<fan::mat4>& transformations) {

        for (int i = 0; i < parsed_model.model_data.vertices[mesh_id].size(); ++i) {

          if (transformations.empty()) {

            fan::vec4 vertex_position = model * fan::vec4(parsed_model.model_data.vertices[mesh_id][i].position, 1.0);
            m_modified_verticies[mesh_id][i].position = fan::vec3(vertex_position.x, vertex_position.y, vertex_position.z);
          }
          else {
            fan::mat4 interpolated_bone_transform = calculate_bone_transform(i, transformations);

            fan::vec4 vertex_position = fan::vec4(parsed_model.model_data.vertices[mesh_id][i].position, 1.0);

            fan::vec4 result = interpolated_bone_transform * vertex_position;

            m_modified_verticies[mesh_id][i].position = fan::vec3(result.x, result.y, result.z);
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

      void get_tpose(animation_data_t& animation, joint_t& joint) {
        animation.pose[joint.id] = joint.global_transform * joint.offset;

        animation.joint_poses[joint.id].position = 0;
        animation.joint_poses[joint.id].scale = 1;
        animation.joint_poses[joint.id].rotation = fan::quat();
        //animation.joint_poses[joint.id].joint_offset = joint.offset;
        for (fan_3d::model::joint_t& child : joint.children) {
          get_tpose(animation, child);
        }
      }

      void fk_get_pose(animation_data_t& animation, joint_t& joint, fan::mat4& parent_transform) {

        dt = fmod(dt, animation.duration);
        fan::vec3 position = 0, scale = 1;
        fan::quat rotation;

        bool not_enough_info = false;

        auto found = animation.bone_transforms.find(joint.name);
        f32_t prev_weight = found->second.weight;
        if (found == animation.bone_transforms.end()) {
          fan::throw_error("invalid bone data");
        }
        else {
          found->second.weight = 1.f;

          not_enough_info = fk_parse_joint_data(found->second, position, rotation, scale);
          if (showing_temp_rot && joint.id == active_joint) {
            rotation = temp_rotations[joint.id];
            not_enough_info = false;
          }

          found->second.weight = prev_weight;
        }

        fan::mat4 local_transform;
        if (not_enough_info) {
          animation.joint_poses[joint.id].position = fan::vec3(0);
          animation.joint_poses[joint.id].rotation = fan::quat();
          animation.joint_poses[joint.id].scale = 1;
          local_transform = joint.local_transform;
        }
        else {
          animation.joint_poses[joint.id].position = position;
          animation.joint_poses[joint.id].scale = scale;
          animation.joint_poses[joint.id].rotation = rotation;
        }
        fan::mat4 global_transform = parent_transform * local_transform;

        for (fan_3d::model::joint_t& child : joint.children) {
          fk_get_pose(animation, child, global_transform);
        }
      }
      void get_pose(animation_data_t& animation, joint_t& joint, fan::mat4& parent_transform) {

        dt = fmod(dt, animation.duration);
        fan::vec3 position = 0, scale = 1;
        fan::quat rotation;

        bool not_enough_info = false;

        auto found = animation.bone_transforms.find(joint.name);
        f32_t prev_weight = found->second.weight;
        if (found == animation.bone_transforms.end()) {
          fan::throw_error("invalid bone data");
        }
        else {
          found->second.weight = 1.f;
          not_enough_info = fk_parse_joint_data(found->second, position, rotation, scale);
          found->second.weight = prev_weight;
        }

        fan::mat4 mtranslation = fan::mat4(1).translate(position);
        fan::mat4 mrotation = fan::mat4(rotation);
        fan::mat4 mscale = fan::mat4(1).scale(scale);

        fan::mat4 local_transform;
        if (not_enough_info) {
          local_transform = mtranslation * mrotation * mscale;
        }
        else {
          local_transform = mtranslation * mrotation * mscale;
        }
        fan::mat4 global_transform = parent_transform * local_transform;

        animation.pose[joint.id] = m_transform * global_transform * joint.offset;

        for (fan_3d::model::joint_t& child : joint.children) {
          get_pose(animation, child, global_transform);
        }
      }


      void interpolate_default_animation(std::vector<fan::mat4>& joint_transforms) {

        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;

        fan::mat4 global_transform;

        joint_transforms = default_animation.pose;
      }

      void fk_interpolate_animations(std::vector<fan::mat4>& joint_transforms, animation_data_t& animation, joint_t& joint, fan::mat4& parent_transform, f32_t animation_weight) {

        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;

        fan::mat4 global_transform;

        if (animation_list.empty()) {
          // tpose
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

          fan::mat4 local_transform = joint.local_transform;
          local_transform = local_transform.translate(position);
          local_transform = local_transform.rotate(rotation);
          local_transform = local_transform.scale(scale);
          joint_transforms[joint.id] = m_transform * parent_transform * local_transform * joint.offset;
          global_transform = parent_transform * local_transform;
        }

        for (fan_3d::model::joint_t& child : joint.children) {
          fk_interpolate_animations(joint_transforms, animation, child, global_transform, animation_weight);
        }
      }

      void calculate_tpose() {
        get_tpose(tpose, parsed_model.model_data.skeleton);
      }
      void calculate_default_pose() {
        fan::mat4 initial(1);
        // not sure if this is supposed to be here
        // blender gave option to flip export axis, but it doesnt seem to flip it in animation
        // but only in tpose
        initial = initial.rotate(-fan::math::pi / 2, fan::vec3(1, 0, 0));
        get_pose(default_animation, parsed_model.model_data.skeleton, initial);
      }
      void calculate_poses() {
        fan::mat4 initial(1);
        // not sure if this is supposed to be here
        // blender gave option to flip export axis, but it doesnt seem to flip it in animation
        // but only in tpose
        initial = initial.rotate(-fan::math::pi / 2, fan::vec3(1, 0, 0));
        calculate_tpose(); // not really needed, can be just called initally
        calculate_default_pose();
        for (auto& i : animation_list) {
          fk_get_pose(i.second, parsed_model.model_data.skeleton, initial);
        }
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
          (*triangles)[i / edge_count].p[i % edge_count] = m_modified_verticies[mesh_id][i].position;
          (*triangles)[i / edge_count].tc[i % edge_count] = m_modified_verticies[mesh_id][i].uv;
        }
      }

      void iterate_joints(joint_t& joint, auto lambda) {
        lambda(joint);
        for (fan_3d::model::joint_t& child : joint.children) {
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

      joint_t* get_joint(const fan::string& name) {
        bool found = false;
        joint_t* pjoint = nullptr;
        iterate_joints(parsed_model.model_data.skeleton,
          [&](joint_t& joint) {
            if (found) {
              return;
            }
            if (name == joint.name) {
              found = true;
              pjoint = &joint;
            }
          });
        return pjoint;
      }
      joint_t* get_joint(int bone_id) {
        return get_joint(bone_strings[bone_id]);
      }

      using anim_key_t = std::string;

      anim_key_t active_anim;
      std::unordered_map<anim_key_t, animation_data_t> animation_list;

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

      fan::string create_an(const fan::string& key, f32_t weight) {
        if (animation_list.empty()) {
          active_anim = key;
        }
        auto& node = animation_list[key];
        node.weight = weight;
        //node = m_animation;
        node.duration = default_animation.duration;
        // initialize with tpose
        node.joint_poses.resize(default_animation.pose.size(), { {}, {}, 1 });
        node.pose.resize(default_animation.pose.size(), fan::mat4(1));
        iterate_joints(parsed_model.model_data.skeleton,
          [&](joint_t& joint) {
            node.bone_transforms[joint.name];
          }
        );

        // user should worry about it
        //for (auto& anim : animation_list) {
        //  anim.second.weight = 1.f / animation_list.size();
        //}
        return key;
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
        //node.bone_transforms = 
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

      uint32_t get_rotation_by_dt(animation_data_t& anim, int joint_id, f32_t dt) {
        auto& bt = anim.bone_transforms[bone_strings[joint_id]];
        auto it = std::upper_bound(bt.rotation_timestamps.begin(), bt.rotation_timestamps.end(), dt);

        int insert_pos = std::distance(bt.rotation_timestamps.begin(), it);
        if (insert_pos) {
          --insert_pos;
        }
        return insert_pos;
      }

      std::vector<fan::mat4> fk_transformations;

      bool toggle_rotate = false;
      bool showing_temp_rot = false;
      std::vector<fan::quat> temp_rotations;

      parsed_model_t parsed_model;
      fan_3d::model::animation_data_t tpose;
      fan_3d::model::animation_data_t default_animation;

      // custom poses
      std::vector<std::vector<fan_3d::model::vertex_t>> m_modified_verticies;

      fan::mat4 m_transform;

      std::vector<fan::string> bone_strings;
      f32_t dt = 0;

      int active_joint = -1;
    };

    struct animator_t {

      std::string animation_vs = R"(
					#version 440 core
					layout (location = 0) in vec3 vertex;
					layout (location = 1) in vec3 normal;
					layout (location = 2) in vec2 uv;
					layout (location = 3) in vec4 bone_ids;
					layout (location = 4) in vec4 bone_weights;
          layout (location = 5) in vec3 vertex1;
          layout (location = 6) in vec3 tangent;  
          layout (location = 7) in vec3 bitangent;  
          layout (location = 8) in vec4 diffuse;  

					out vec2 tex_coord;
					out vec3 v_normal;
					out vec3 v_pos;
					out vec4 bw;
          out vec4 v_diffuse;

          out vec3 c_tangent;
          out vec3 c_bitangent;
          //out vec3 c_bitangent ;

					uniform mat4 projection;
					uniform mat4 view;
          uniform int curtains;

					void main()
					{
            mat4 model = mat4(1);
            //if (curtains == 0) {
              //model[0][0] = 0.0001;
              //model[1][1] = 0.0001;
              //model[2][2] = 0.0001;
            //}
            //if (curtains == 0) {
              model[0][0] = 0.01;
              model[1][1] = 0.01;
              model[2][2] = 0.01;
            //}

						gl_Position = projection * view * model * vec4(vertex, 1.0);
            tex_coord = uv;
            v_pos = vec3(model * vec4(vertex, 1.0));
						v_normal = mat3(transpose(inverse(model))) * normal;
						v_normal = normalize(v_normal);
            c_tangent = tangent;
            c_bitangent = bitangent;
            //v_bitangent = cross(normal, tangent);
            v_diffuse = diffuse;
					}
			)";

      std::string animation_gpu_vs = R"(
					#version 440 core
					layout (location = 0) in vec3 vertex;
					layout (location = 1) in vec3 normal;
					layout (location = 2) in vec2 uv;
					layout (location = 3) in vec4 bone_ids;
					layout (location = 4) in vec4 bone_weights;
          layout (location = 5) in vec3 vertex1;
          layout (location = 6) in vec3 tangent;  
          layout (location = 7) in vec3 bitangent;  
          layout (location = 8) in vec4 diffuse;  

					out vec2 tex_coord;
					out vec3 v_normal;
					out vec3 v_pos;
					out vec4 bw;
          out vec4 v_diffuse;

          out vec3 c_tangent;
          out vec3 c_bitangent;
          //out vec3 c_bitangent ;

					uniform mat4 projection;
					uniform mat4 view;
          uniform int curtains;
          uniform mat4 model;

					void main()
					{
						gl_Position = projection * view * model * vec4(vertex, 1.0);
            tex_coord = uv;
            v_pos = vec3(model * vec4(vertex, 1.0));
						v_normal = mat3(transpose(inverse(model))) * normal;
						v_normal = normalize(v_normal);
            c_tangent = tangent;
            c_bitangent = bitangent;
            //v_bitangent = cross(normal, tangent);
            v_diffuse = diffuse;
					}
			)";


      std::string animation_fs = R"(
      #version 440 core

     in vec2 tex_coord;
     in vec3 v_normal;
     in vec3 v_pos;
     in vec4 bw;
     in vec4 v_diffuse;

     in vec3 c_tangent;
     in vec3 c_bitangent;
     layout (location = 0) out vec4 color; 

     uniform sampler2D diff_texture;
     uniform sampler2D norm_texture;
     uniform sampler2D roughness_texture;

     void main()
     {
	      vec3 albedo = texture(diff_texture, tex_coord).rgb;
        color = vec4(v_diffuse);
     }

			)";

      void init_render_object(uint32_t i) {
        auto& context = gloco->get_context();
        render_object_t& ro = render_objects[i];
        ro.vao.open(context);
        ro.vbo.open(context, fan::opengl::GL_ARRAY_BUFFER);
        ro.vao.bind(context);
        upload_modified_vertices(i);

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
        gloco->get_context().opengl.glEnableVertexAttribArray(8);
        gloco->get_context().opengl.glVertexAttribPointer(8, 4, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, sizeof(fan_3d::model::vertex_t), (fan::opengl::GLvoid*)offsetof(fan_3d::model::vertex_t, diffuse));
        gloco->get_context().opengl.glBindVertexArray(0);
      }

      animator_t(const fan::string& path, bool gpu = false) : fms(path) {

        //fms.iterate_joints(fms.parsed_model.model_data.skeleton, [this](const joint_t& joint) {
        //  loco_t::shapes_t::rectangle_3d_t::properties_t rp;
        //  rp.size = 0.1;
        //  rp.color = fan::colors::red;
        //  rp.position = joint.global_transform.get_translation();
        //  shapes.push_back(rp);
        //  });

       /* for (int i = 0; i < 10; ++i) {
          loco_t::shapes_t::rectangle_3d_t::properties_t rp;
          rp.size = fan::random::value_f32(0.01, 0.5);
          rp.color = fan::colors::red;
          rp.position = fan::vec3(-20, 0, i);
          shapes.push_back(rp);
        }*/
        //image.load(fms.parsed_model.texture_data.diffuse_texture_path_list[0]);
        //fms.parsed_model.texture_data.
        m_shader.open();
        if (gpu) {
          m_shader.set_vertex(
            animation_gpu_vs
          );
        }
        else {
          m_shader.set_vertex(
            animation_vs
          );
        }

        m_shader.set_fragment(
          animation_fs
        );

        m_shader.compile();

        auto& context = gloco->get_context();
        render_objects.resize(fms.parsed_model.model_data.vertices.size());
        for (int i = 0; i < render_objects.size(); ++i) {
          
          init_render_object(i);

          if (fms.parsed_model.texture_names[i].diffuse.size()) {
            auto found = diffuse_images.find(fms.parsed_model.texture_names[i].diffuse);
            if (found == diffuse_images.end()) {
              diffuse_images[fms.parsed_model.texture_names[i].diffuse].load(fms.parsed_model.texture_names[i].diffuse);
            }
          }
          if (fms.parsed_model.texture_names[i].normal.size()) {
            auto found = normal_images.find(fms.parsed_model.texture_names[i].normal);
            if (found == normal_images.end()) {
              normal_images[fms.parsed_model.texture_names[i].normal].load(fms.parsed_model.texture_names[i].normal);
            }
          }
          if (fms.parsed_model.texture_names[i].roughness.size()) {
            auto found = roughness_images.find(fms.parsed_model.texture_names[i].roughness);
            if (found == roughness_images.end()) {
              roughness_images[fms.parsed_model.texture_names[i].roughness].load(fms.parsed_model.texture_names[i].roughness);
            }
          }
          if (fms.parsed_model.texture_names[i].metallic.size()) {
            auto found = metallic_images.find(fms.parsed_model.texture_names[i].metallic);
            if (found == metallic_images.end()) {
              metallic_images[fms.parsed_model.texture_names[i].metallic].load(fms.parsed_model.texture_names[i].metallic);
            }
          }
        }
      }

      void upload_modified_vertices(uint32_t i) {
        render_objects[i].vao.bind(gloco->get_context());

        render_objects[i].vbo.write_buffer(
          gloco->get_context(),
          &fms.m_modified_verticies[i][0],
          sizeof(fan_3d::model::vertex_t) * fms.m_modified_verticies[i].size()
        );
      }

      void draw(bool curtain) {
        m_shader.use();

        fan::mat4 projection(1);
        static constexpr f32_t fov = 90.f;
        projection = fan::math::perspective<fan::mat4>(fan::math::radians(fov), (f32_t)gloco->window.get_size().x / (f32_t)gloco->window.get_size().y, 0.1f, 1000.0f);
        fan::mat4 view(gloco->default_camera_3d->camera.get_view_matrix());

        m_shader.set_mat4("projection", projection);
        m_shader.set_mat4("view", view);



        gloco->get_context().set_depth_test(true);
        m_shader.set_int("diff_texture", 0);
        m_shader.set_int("norm_texture", 1);
        m_shader.set_int("roughness_texture", 2);
        m_shader.set_int("metallic_texture", 4);
        static fan::vec3 light_pos = 0;
        ImGui::Text(gloco->default_camera_3d->camera.position.to_string().c_str());
        ImGui::DragFloat3("light position", light_pos.data());
        fan::vec4 lpt = fan::vec4(light_pos, 1);
        m_shader.set_vec3("light_pos", *(fan::vec3*)&lpt);


        static f32_t f0 = 0;
        ImGui::DragFloat("f0", &f0, 0.001, 0, 1);
        m_shader.set_float("F0", f0);


        static f32_t metallic = 0;
        ImGui::DragFloat("metallic", &metallic, 0.001, 0, 1);
        m_shader.set_float("metallic", metallic);

        static f32_t roughness = 0;
        ImGui::DragFloat("rough", &roughness, 0.001, 0, 1);
        m_shader.set_float("rough", roughness);

        m_shader.set_int("curtains", curtain);
       // shapes[0].set_position(light_pos);

        static f32_t light_intensity = 1;
        ImGui::DragFloat("light_intensity", &light_intensity, 0.1);
        m_shader.set_float("light_intensity", light_intensity); 

        gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE3);
        gloco->get_context().opengl.glBindTexture(fan::opengl::GL_TEXTURE_CUBE_MAP, envMapTexture);
        m_shader.set_int("envMap", 3);

        auto& context = gloco->get_context();
        context.opengl.glDisable(fan::opengl::GL_BLEND);
        //context.opengl.call(context.opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
        for (int i = 0; i < render_objects.size(); ++i) {

          m_shader.set_mat4("model", render_objects[i].m); // only for gpu vs

          render_objects[i].vao.bind(context);
          m_shader.set_mat4("transform", fms.parsed_model.transforms[i]);
          fan::vec4 cpt = fms.parsed_model.transforms[i] * fan::vec4(gloco->default_camera_3d->camera.position, 1);
          m_shader.set_vec3("view_p", gloco->default_camera_3d->camera.position);
          if (i < fms.parsed_model.texture_names.size()) {
            if (fms.parsed_model.texture_names[i].diffuse.size()) {
              gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
              diffuse_images[fms.parsed_model.texture_names[i].diffuse].bind_texture();
              m_shader.set_bool("has_texture", 1);
            }
            else {
              m_shader.set_bool("has_texture", 0);
            }
            if (fms.parsed_model.texture_names[i].normal.size()) {
              gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
              normal_images[fms.parsed_model.texture_names[i].normal].bind_texture();
              m_shader.set_bool("has_normal", 1);
            }
            if (fms.parsed_model.texture_names[i].roughness.size()) {
              gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE2);
              roughness_images[fms.parsed_model.texture_names[i].roughness].bind_texture();
            }
            if (fms.parsed_model.texture_names[i].metallic.size()) {
              gloco->get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE4);
              metallic_images[fms.parsed_model.texture_names[i].metallic].bind_texture();
            }
          }

          /*render_objects[i].vbo.write_buffer(
            gloco->get_context(),
            &fms.m_modified_verticies[i][0],
            sizeof(fan_3d::model::vertex_t) * fms.m_modified_verticies[i].size()
          );*/

          gloco->get_context().opengl.glDrawArrays(fan::opengl::GL_TRIANGLES, 0, fms.m_modified_verticies[i].size());
        }
      }


      void display_animations() {
        for (auto& anim_pair : fms.animation_list) {
          bool nodeOpen = ImGui::TreeNode(anim_pair.first.c_str());
          if (nodeOpen) {
            auto& anim = anim_pair.second;
            bool time_stamps_open = ImGui::TreeNode("timestamps");
            if (time_stamps_open) {
              fms.iterate_joints(fms.parsed_model.model_data.skeleton, [&](joint_t& joint) {
                auto& bt = anim.bone_transforms[joint.name];
                uint32_t data_count = bt.rotation_timestamps.size();
                if (data_count) {
                  bool node_open = ImGui::TreeNode(joint.name.c_str());
                  if (node_open) {
                    for (int i = 0; i < bt.rotation_timestamps.size(); ++i) {
                      ImGui::DragFloat(
                        ("rotation:" + std::to_string(i)).c_str(),
                        &anim.bone_transforms[joint.name].rotation_timestamps[i]
                      );
                    }
                    ImGui::TreePop();
                  }
                }
                });
              ImGui::TreePop(); // Close the "timestamps" node here
            }
            bool properties_open = ImGui::TreeNode("properties"); // Now "properties" will be outside of "timestamps"
            if (properties_open) {
              fms.iterate_joints(fms.parsed_model.model_data.skeleton, [&](joint_t& joint) {
                auto& bt = anim.bone_transforms[joint.name];
                uint32_t data_count = bt.rotations.size();
                if (data_count) {
                  bool node_open = ImGui::TreeNode(joint.name.c_str());
                  if (node_open) {
                    for (int i = 0; i < bt.rotations.size(); ++i) {
                      ImGui::DragFloat4(
                        ("rotation:" + std::to_string(i)).c_str(),
                        anim.bone_transforms[joint.name].rotations[i].data()
                      );
                    }
                    ImGui::TreePop();
                  }
                }
                });
              ImGui::TreePop(); // Close the "properties" node here
            }
            ImGui::TreePop();
          }
        }
      }

      void mouse_modify_joint() {

        ImGui::DragFloat("current time", &fms.dt, 1, 0, fms.default_animation.duration);
        ImGui::Text(fan::format("camera pos: {}\ntotal time: {:.2f}", gloco->default_camera_3d->camera.position, fms.default_animation.duration).c_str());
        static bool play = false;
        if (ImGui::Checkbox("play animation", &play)) {
          fms.showing_temp_rot = false;
        }
        static int x = 0;
        static fan::time::clock c;
        if (play) {
          if (x == 0) {
            c.start();
            x++;
          }
          fms.dt = c.elapsed() / 1e+6;
        }


        int current_id = fms.get_active_animation_id();
        std::vector<const char*> animations;
        for (auto& i : fms.animation_list) {
          animations.push_back(i.first.c_str());
        }
        if (ImGui::ListBox("animation list", &current_id, animations.data(), animations.size())) {
          fms.active_anim = animations[current_id];
        }
        ImGui::DragFloat("animation weight", &fms.animation_list[fms.active_anim].weight, 0.01, 0, 1);

        if (fms.active_joint != -1) {
          auto& anim = fms.get_active_animation();
          auto& bt = anim.bone_transforms[fms.bone_strings[fms.active_joint]];
          for (int i = 0; i < bt.rotations.size(); ++i) {
            ImGui::DragFloat4(("rotations:" + std::to_string(i)).c_str(), bt.rotations[i].data(), 0.01);
          }

          static int32_t current_frame = 0;
          if (!play) {
            fms.dt = current_frame;
          }
          else {
            current_frame = fmodf(c.elapsed() / 1e+6 / 4, anim.duration);
          }

          static f32_t prev_frame = 0;
          if (prev_frame != fms.dt) {
            fms.showing_temp_rot = false;
            prev_frame = fms.dt;
          }

          if (ImGui::Button("save keyframe")) {
            fms.fk_set_rot(fms.active_anim, fms.bone_strings[fms.active_joint], current_frame / 1000.f, anim.joint_poses[fms.active_joint].rotation);
          }

          //fan::print(current_frame);
          int32_t startFrame = 0;
          int32_t endFrame = std::ceil(fms.default_animation.duration);
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

      void set_active_joint(int joint_id) {
        fms.active_joint = joint_id;

        joint_t* joint = fms.get_joint(joint_id);

        for (int i = 0; i < std::size(joint_controls); ++i) {
          loco_t::shapes_t::rectangle_3d_t::properties_t rp;
          rp.size = fan::vec3(0.1, 0.5, 0.1);
          rp.color = std::to_array({ fan::colors::red, fan::colors::green, fan::colors::blue })[i];
          rp.position =
            std::to_array({
            fan::vec3(1, 0, 0),
            fan::vec3(0, 0, 1),
            fan::vec3(-1, 0, 0)
              })[i]
            + joint->global_transform.get_translation();
          joint_controls[i] = rp;
        }
      }


      fms_t fms;

      loco_t::shader_t m_shader;
      struct render_object_t {
        fan::opengl::core::vao_t vao;
        fan::opengl::core::vbo_t vbo;
        fan::mat4 m{ 1 };
      };
      std::unordered_map<std::string, loco_t::image_t> diffuse_images;
      std::unordered_map<std::string, loco_t::image_t> normal_images;
      std::unordered_map<std::string, loco_t::image_t> roughness_images;
      std::unordered_map<std::string, loco_t::image_t> metallic_images;
      std::vector<render_object_t> render_objects;
      std::vector<loco_t::shape_t> shapes;

      static constexpr uint8_t axis_count = 3;
      loco_t::shape_t joint_controls[axis_count];
      fan::opengl::GLuint envMapTexture;
    };
  }
}