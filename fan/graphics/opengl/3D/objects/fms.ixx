module;

#include <fan/utility.h>

#if defined(FAN_3D)

#include <fan/types/dme.h>

#include <vector>
#include <cassert>
#include <locale>
#include <set>

#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

//#define STB_IMAGE_IMPLEMENTATION
#include <fan/stb/stb_image.h>

#endif

export module fan.graphics.fms;

#if defined(FAN_3D)

export import fan.types.matrix;
export import fan.print;
export import fan.graphics;

import fan.graphics.gui.base;


export namespace fan {

  struct texture_type {
    enum {
      none = aiTextureType_NONE,
      diffuse = aiTextureType_DIFFUSE,
      specular = aiTextureType_SPECULAR,
      ambient = aiTextureType_AMBIENT,
      emissive = aiTextureType_EMISSIVE,
      height = aiTextureType_HEIGHT,
      normals = aiTextureType_NORMALS,
      shininess = aiTextureType_SHININESS,
      opacity = aiTextureType_OPACITY,
      displacement = aiTextureType_DISPLACEMENT,
      lightmap = aiTextureType_LIGHTMAP,
      reflection = aiTextureType_REFLECTION,
      base_color = aiTextureType_BASE_COLOR,
      normal_camera = aiTextureType_NORMAL_CAMERA,
      emission_color = aiTextureType_EMISSION_COLOR,
      metalness = aiTextureType_METALNESS,
      diffuse_roughness = aiTextureType_DIFFUSE_ROUGHNESS,
      ambient_occlusion = aiTextureType_AMBIENT_OCCLUSION,
      unknown = aiTextureType_UNKNOWN,
      sheen = aiTextureType_SHEEN,
      clearcoat = aiTextureType_CLEARCOAT,
      transmission = aiTextureType_TRANSMISSION,
      //maya_base = aiTextureType_MAYA_BASE,
      //maya_specular = aiTextureType_MAYA_SPECULAR,
      //maya_specular_color = aiTextureType_MAYA_SPECULAR_COLOR,
      //maya_specular_roughness = aiTextureType_MAYA_SPECULAR_ROUGHNESS,
      //anisotropy = aiTextureType_ANISOTROPY,
      //gltf_metallic_roughness = aiTextureType_GLTF_METALLIC_ROUGHNESS
    };
  };


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
      
      //fan::vec3 user_position = 0;
      //fan::vec3 user_rotation = 0;
      //fan::vec3 user_scale = 1;
      fan::mat4 user_transform{1};

      // this appears to be different than transformation
      fan::mat4 get_local_matrix() const {
        return fan::mat4(1).translate(position) * fan::mat4(1).rotate(rotation) * fan::mat4(1).scale(scale);
      }
    };
    inline constexpr auto texture_max = AI_TEXTURE_TYPE_MAX + 1;
    struct mesh_t {
      std::vector<fan::model::vertex_t> vertices;
      std::vector<uint32_t> indices;
      uint32_t indices_len = 0;

      std::string texture_names[texture_max]{};
    };
    // pm -- parsed model
    struct pm_texture_data_t {
      fan::vec2ui size = 0;
      std::vector<uint8_t> data;
      int channels = 0;
    };
    std::unordered_map<std::string, pm_texture_data_t> cached_texture_data;
    struct pm_material_data_t {
      fan::vec4 color[AI_TEXTURE_TYPE_MAX + 1];
    };
    // fan model stuff
    struct fms_t {
      struct properties_t {
        std::string path;
        std::string texture_path = "models/textures";
        int use_cpu = false;
      };
      fms_t() = default;
      fms_t(const properties_t& fmi) {
        p = fmi;
        if (!load_model(fmi.path)) {
          fan::throw_error("failed to load model:" + fmi.path);
        }
       // importer.~Importer();
      }

      // ---------------------model hierarchy---------------------

      void process_node(aiNode* node, const aiMatrix4x4& parent, fan::vec3& out_min, fan::vec3& out_max) {
        aiMatrix4x4 global = parent * node->mTransformation;

        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
          aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

          fan::vec3 mesh_min(FLT_MAX), mesh_max(-FLT_MAX);

          mesh_t m = process_mesh(mesh, global, mesh_min, mesh_max);


          out_min = std::min(out_min, mesh_min);
          out_max = std::max(out_max, mesh_max);

          pm_material_data_t mat = load_materials(mesh);
          material_data_vector.push_back(mat);
          load_textures(m, mesh);
          process_bone_offsets(mesh);

          meshes.push_back(m);
        }

        for (unsigned int i = 0; i < node->mNumChildren; i++) {
          process_node(node->mChildren[i], global, out_min, out_max);
        }
      }

      struct edge_key {
        uint32_t a;
        uint32_t b;

        bool operator==(const edge_key &o) const {
          return a == o.a && b == o.b;
        }
      };

      struct edge_key_hash {
        size_t operator()(const edge_key &k) const {
          return (size_t(k.a) << 32) ^ size_t(k.b);
        }
      };

      struct tri_edge_ref {
        uint32_t tri_index;
        uint8_t edge_slot;
      };

      static f32_t tri_uv_error(const mesh_t &mesh, uint32_t i0, uint32_t i1, uint32_t i2) {
        const auto &v0 = mesh.vertices[i0];
        const auto &v1 = mesh.vertices[i1];
        const auto &v2 = mesh.vertices[i2];

        fan::vec2 e01 = v1.uv - v0.uv;
        fan::vec2 e12 = v2.uv - v1.uv;
        fan::vec2 e20 = v0.uv - v2.uv;

        return e01.dot(e01) + e12.dot(e12) + e20.dot(e20);
      }

      static bool triangles_coplanar(const mesh_t &mesh,
        uint32_t i0, uint32_t i1, uint32_t i2,
        uint32_t j0, uint32_t j1, uint32_t j2,
        f32_t cos_threshold = 0.99f
      ) {
        const auto &a0 = mesh.vertices[i0];
        const auto &a1 = mesh.vertices[i1];
        const auto &a2 = mesh.vertices[i2];

        const auto &b0 = mesh.vertices[j0];
        const auto &b1 = mesh.vertices[j1];
        const auto &b2 = mesh.vertices[j2];

        fan::vec3 na = (a1.position - a0.position).cross(a2.position - a0.position);
        fan::vec3 nb = (b1.position - b0.position).cross(b2.position - b0.position);

        f32_t la2 = na.dot(na);
        f32_t lb2 = nb.dot(nb);
        if (la2 == 0.0f || lb2 == 0.0f) {
          return false;
        }

        f32_t c = na.dot(nb) / std::sqrt(la2 * lb2);
        return c > cos_threshold;
      }

      static bool segments_intersect(
        const fan::vec2& p0,
        const fan::vec2& p1,
        const fan::vec2& p2,
        const fan::vec2& p3
      )
      {
        auto cross = [](const fan::vec2& a, const fan::vec2& b) {
          return a.x * b.y - a.y * b.x;
        };

        fan::vec2 r = p1 - p0;
        fan::vec2 s = p3 - p2;

        float denom = cross(r, s);
        float numer1 = cross(p2 - p0, r);
        float numer2 = cross(p2 - p0, s);

        if (denom == 0.0f) {
          return false;
        }

        float t = numer1 / denom;
        float u = numer2 / denom;

        return t > 0.0f && t < 1.0f && u > 0.0f && u < 1.0f;
      }


      bool is_valid_quad(const mesh_t& mesh, uint32_t q[4]) {
        const vec3& p0 = mesh.vertices[q[0]].position;
        const vec3& p1 = mesh.vertices[q[1]].position;
        const vec3& p2 = mesh.vertices[q[2]].position;
        const vec3& p3 = mesh.vertices[q[3]].position;

        vec3 n = (p1 - p0).cross(p2 - p0);
        if (n.dot((p2 - p1).cross(p3 - p1)) < 0) return false;
        if (n.dot((p3 - p2).cross(p0 - p2)) < 0) return false;

        if (segments_intersect(p0, p1, p2, p3)) return false;
        if (segments_intersect(p1, p2, p3, p0)) return false;

        return true;
      }


      void fix_uv_diagonals(mesh_t &mesh) {
        auto &indices = mesh.indices;
        auto &verts = mesh.vertices;

        if (indices.size() % 3 != 0 || verts.empty()) {
          return;
        }

        std::unordered_map<edge_key, tri_edge_ref, edge_key_hash> edge_map;
        edge_map.reserve(indices.size());

        size_t tri_count = indices.size() / 3;

        for (uint32_t t = 0; t < tri_count; t++) {
          uint32_t i0 = indices[t * 3 + 0];
          uint32_t i1 = indices[t * 3 + 1];
          uint32_t i2 = indices[t * 3 + 2];

          uint32_t tri_idx[3] = { i0, i1, i2 };

          for (uint8_t e = 0; e < 3; e++) {
            uint32_t a = tri_idx[e];
            uint32_t b = tri_idx[(e + 1) % 3];

            edge_key key;
            key.a = fan::math::min(a, b);
            key.b = fan::math::max(a, b);

            auto it = edge_map.find(key);
            if (it == edge_map.end()) {
              tri_edge_ref ref;
              ref.tri_index = t;
              ref.edge_slot = e;
              edge_map.insert({ key, ref });
            }
            else {
              uint32_t t2 = it->second.tri_index;
              if (t2 == t) {
                continue;
              }

              uint32_t j0 = indices[t2 * 3 + 0];
              uint32_t j1 = indices[t2 * 3 + 1];
              uint32_t j2 = indices[t2 * 3 + 2];

              uint32_t u[6] = { i0, i1, i2, j0, j1, j2 };
              uint32_t quad[4];
              uint32_t qc = 0;

              for (uint32_t k = 0; k < 6; k++) {
                uint32_t v = u[k];
                bool found = false;
                for (uint32_t m = 0; m < qc; m++) {
                  if (quad[m] == v) {
                    found = true;
                    break;
                  }
                }
                if (!found) {
                  quad[qc++] = v;
                  if (qc > 4) {
                    break;
                  }
                }
              }

              if (!is_valid_quad(mesh, quad)) {
                continue;
              }

              if (qc != 4) {
                continue;
              }

              if (!triangles_coplanar(mesh, i0, i1, i2, j0, j1, j2, 0.99f)) {
                continue;
              }

              fan::vec3 c = (verts[quad[0]].position +
                verts[quad[1]].position +
                verts[quad[2]].position +
                verts[quad[3]].position) * 0.25f;

              f32_t ang[4];
              for (uint32_t k = 0; k < 4; k++) {
                fan::vec3 d = verts[quad[k]].position - c;
                ang[k] = std::atan2(d.y, d.x);
              }

              for (uint32_t a0 = 0; a0 < 4; a0++) {
                for (uint32_t b0 = a0 + 1; b0 < 4; b0++) {
                  if (ang[b0] < ang[a0]) {
                    f32_t ta = ang[a0];
                    ang[a0] = ang[b0];
                    ang[b0] = ta;
                    uint32_t tv = quad[a0];
                    quad[a0] = quad[b0];
                    quad[b0] = tv;
                  }
                }
              }

              uint32_t q0 = quad[0];
              uint32_t q1 = quad[1];
              uint32_t q2 = quad[2];
              uint32_t q3 = quad[3];

              uint32_t c0 = i0, c1 = i1, c2 = i2;
              uint32_t c3 = j0, c4 = j1, c5 = j2;

              f32_t uv_current =
                tri_uv_error(mesh, c0, c1, c2) +
                tri_uv_error(mesh, c3, c4, c5);

              fan::vec3 g01 = verts[c1].position - verts[c0].position;
              fan::vec3 g12 = verts[c2].position - verts[c1].position;
              fan::vec3 g20 = verts[c0].position - verts[c2].position;
              fan::vec3 g34 = verts[c4].position - verts[c3].position;
              fan::vec3 g45 = verts[c5].position - verts[c4].position;
              fan::vec3 g53 = verts[c3].position - verts[c5].position;

              f32_t geom_current =
                g01.dot(g01) + g12.dot(g12) + g20.dot(g20) +
                g34.dot(g34) + g45.dot(g45) + g53.dot(g53);

              f32_t total_current = uv_current + geom_current * 0.000001f;

              f32_t uv_a =
                tri_uv_error(mesh, q0, q1, q2) +
                tri_uv_error(mesh, q0, q2, q3);

              fan::vec3 a01 = verts[q1].position - verts[q0].position;
              fan::vec3 a12 = verts[q2].position - verts[q1].position;
              fan::vec3 a20 = verts[q0].position - verts[q2].position;
              fan::vec3 a03 = verts[q3].position - verts[q0].position;
              fan::vec3 a32 = verts[q2].position - verts[q3].position;
              fan::vec3 a23 = verts[q3].position - verts[q2].position;

              f32_t geom_a =
                a01.dot(a01) + a12.dot(a12) + a20.dot(a20) +
                a03.dot(a03) + a32.dot(a32) + a23.dot(a23);

              f32_t total_a = uv_a + geom_a * 0.000001f;

              f32_t uv_b =
                tri_uv_error(mesh, q0, q1, q3) +
                tri_uv_error(mesh, q1, q2, q3);

              fan::vec3 b01 = verts[q1].position - verts[q0].position;
              fan::vec3 b13 = verts[q3].position - verts[q1].position;
              fan::vec3 b30 = verts[q0].position - verts[q3].position;
              fan::vec3 b12 = verts[q2].position - verts[q1].position;
              fan::vec3 b23 = verts[q3].position - verts[q2].position;
              fan::vec3 b31 = verts[q1].position - verts[q3].position;

              f32_t geom_b =
                b01.dot(b01) + b13.dot(b13) + b30.dot(b30) +
                b12.dot(b12) + b23.dot(b23) + b31.dot(b31);

              f32_t total_b = uv_b + geom_b * 0.000001f;

              f32_t best = total_current;
              uint32_t bt0[3] = { c0, c1, c2 };
              uint32_t bt1[3] = { c3, c4, c5 };

              if (total_a + 1e-7f < best) {
                best = total_a;
                bt0[0] = q0; bt0[1] = q1; bt0[2] = q2;
                bt1[0] = q0; bt1[1] = q2; bt1[2] = q3;
              }

              if (total_b + 1e-7f < best) {
                best = total_b;
                bt0[0] = q0; bt0[1] = q1; bt0[2] = q3;
                bt1[0] = q1; bt1[1] = q2; bt1[2] = q3;
              }

              if (best < total_current - 1e-7f) {
                indices[t * 3 + 0] = bt0[0];
                indices[t * 3 + 1] = bt0[1];
                indices[t * 3 + 2] = bt0[2];

                indices[t2 * 3 + 0] = bt1[0];
                indices[t2 * 3 + 1] = bt1[1];
                indices[t2 * 3 + 2] = bt1[2];
              }
            }
          }
        }
      }


      mesh_t process_mesh(aiMesh* mesh, const aiMatrix4x4& transform, fan::vec3& out_min, fan::vec3& out_max) {
        mesh_t new_mesh;
        new_mesh.vertices.resize(mesh->mNumVertices);

        aiMatrix3x3 normalMatrix = aiMatrix3x3(transform);
        normalMatrix = normalMatrix.Transpose().Inverse();

        for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
          vertex_t v;

          aiVector3D p = mesh->mVertices[i];
          p = transform * p;
          v.position = fan::vec3(p.x, p.y, p.z);

          out_min.x = std::min(out_min.x, v.position.x);
          out_min.y = std::min(out_min.y, v.position.y);
          out_min.z = std::min(out_min.z, v.position.z);

          out_max.x = std::max(out_max.x, v.position.x);
          out_max.y = std::max(out_max.y, v.position.y);
          out_max.z = std::max(out_max.z, v.position.z);

          if (mesh->HasNormals()) {
            aiVector3D n = mesh->mNormals[i];
            n = normalMatrix * n;
            v.normal = fan::vec3(n.x, n.y, n.z).normalized();
          }
          else {
            v.normal = fan::vec3(0, 1, 0);
          }

          if (mesh->mTextureCoords[0]) {
            v.uv = fan::vec2(mesh->mTextureCoords[0][i].x,
              mesh->mTextureCoords[0][i].y);
          }
          else {
            v.uv = fan::vec2(0.0f);
          }

          v.bone_ids = fan::vec4i(-1);
          v.bone_weights = fan::vec4(0.0f);

          if (mesh->HasVertexColors(0)) {
            v.color = fan::vec4(mesh->mColors[0][i].r,
              mesh->mColors[0][i].g,
              mesh->mColors[0][i].b,
              mesh->mColors[0][i].a);
          }
          else {
            v.color = fan::vec4(1.0f);
          }

          new_mesh.vertices[i] = v;
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
            float weight = bone->mWeights[j].mWeight;

            int min_index = 0;
            float min_weight = new_mesh.vertices[vertexId].bone_weights[0];

            for (int k = 1; k < 4; k++) {
              if (new_mesh.vertices[vertexId].bone_weights[k] < min_weight) {
                min_weight = new_mesh.vertices[vertexId].bone_weights[k];
                min_index = k;
              }
            }

            if (weight > min_weight) {
              new_mesh.vertices[vertexId].bone_weights[min_index] = weight;
              new_mesh.vertices[vertexId].bone_ids[min_index] = boneId;
            }
          }
        }

        for (auto& v : new_mesh.vertices) {
          float sum = v.bone_weights.x + v.bone_weights.y +
            v.bone_weights.z + v.bone_weights.w;
          if (sum > 0.0f) {
            v.bone_weights /= sum;
          }
        }

        for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
          aiFace& face = mesh->mFaces[i];
          for (uint32_t j = 0; j < face.mNumIndices; j++) {
            new_mesh.indices.push_back(face.mIndices[j]);
          }
        }

        new_mesh.indices_len = new_mesh.indices.size();
        fix_uv_diagonals(new_mesh);

        return new_mesh;
      }

      void load_textures(mesh_t& mesh, aiMesh* ai_mesh) {
        if (scene->mNumMaterials == 0) {
          return;
        }
        aiMaterial* material = scene->mMaterials[ai_mesh->mMaterialIndex];
        for (uint32_t texture_type = 0; texture_type < AI_TEXTURE_TYPE_MAX + 1; ++texture_type) {
          aiString path;
          if (material->GetTexture((aiTextureType)texture_type, 0, &path) != AI_SUCCESS) {
            continue;
          }

          auto embedded_texture = scene->GetEmbeddedTexture(path.C_Str());

          if (embedded_texture && embedded_texture->mHeight == 0) {
            int width, height, nr_channels;
            unsigned char* data = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(embedded_texture->pcData), embedded_texture->mWidth, &width, &height, &nr_channels, 0);
            if (data == nullptr) {
              fan::print("failed to load texture");
              continue;
            }

            // must not collide with other names
            std::string generated_str = path.C_Str() + std::to_string(texture_type);
            mesh.texture_names[texture_type] = generated_str;
            auto& td = fan::model::cached_texture_data[generated_str];
            td.size = fan::vec2(width, height);
            td.data.insert(td.data.end(), data, data + td.size.multiply() * nr_channels);
            td.channels = nr_channels;
            stbi_image_free(data);
          }
          else {
            std::string file_path = p.texture_path + "/" + scene->GetShortFilename(path.C_Str());
            mesh.texture_names[texture_type] = file_path;
            auto found = cached_texture_data.find(file_path);
            if (found == cached_texture_data.end()) {
              fan::image::info_t ii;
              if (fan::image::load(file_path, &ii)) {
                continue;
              }
              auto& td = cached_texture_data[file_path];
              td.size = ii.size;
              td.data.insert(td.data.end(), (uint8_t*)ii.data, (uint8_t*)ii.data + ii.size.multiply() * ii.channels);
              td.channels = ii.channels;
              fan::image::free(&ii);
            }
          }
        }
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

        // Try multiple methods to get base color
        bool found_color = false;
        aiColor4D base_color(1.0f, 1.0f, 1.0f, 1.0f);

        // Method 1: Try AI_MATKEY_BASE_COLOR (newer Assimp)
        if (cmaterial->Get(AI_MATKEY_BASE_COLOR, base_color) == AI_SUCCESS) {
          material_data.color[aiTextureType_DIFFUSE] = fan::vec4(base_color.r, base_color.g, base_color.b, base_color.a);
          fan::print("Found base color via AI_MATKEY_BASE_COLOR:", base_color.r, base_color.g, base_color.b, base_color.a);
          found_color = true;
        }

        // Method 2: Try direct property lookup for glTF
        if (!found_color) {
          if (cmaterial->Get("$clr.base", 0, 0, base_color) == AI_SUCCESS) {
            material_data.color[aiTextureType_DIFFUSE] = fan::vec4(base_color.r, base_color.g, base_color.b, base_color.a);
            fan::print("Found base color via $clr.base:", base_color.r, base_color.g, base_color.b, base_color.a);
            found_color = true;
          }
        }

        // Method 3: Iterate through all properties to find baseColorFactor
        if (!found_color) {
          for (unsigned int i = 0; i < cmaterial->mNumProperties; i++) {
            aiMaterialProperty* prop = cmaterial->mProperties[i];
            std::string key = prop->mKey.C_Str();

            // Debug: print all properties
            fan::print("Material property [", i, "]:", key, "type:", prop->mType, "dataLen:", prop->mDataLength);

            // Look for base color property
            if (key.find("base") != std::string::npos || 
              key.find("Base") != std::string::npos ||
              key.find("COLOR") != std::string::npos) {

              if (prop->mType == aiPTI_Float && prop->mDataLength >= 16) {
                // It's likely a vec4 color
                float* color_data = (float*)prop->mData;
                base_color.r = color_data[0];
                base_color.g = color_data[1];
                base_color.b = color_data[2];
                base_color.a = color_data[3];

                material_data.color[aiTextureType_DIFFUSE] = fan::vec4(base_color.r, base_color.g, base_color.b, base_color.a);
                fan::print("Found base color via property search:", base_color.r, base_color.g, base_color.b, base_color.a);
                found_color = true;
                break;
              }
            }
          }
        }

        // Method 4: Fall back to traditional diffuse color
        if (!found_color) {
          if (cmaterial->Get(AI_MATKEY_COLOR_DIFFUSE, base_color) == AI_SUCCESS) {
            material_data.color[aiTextureType_DIFFUSE] = fan::vec4(base_color.r, base_color.g, base_color.b, base_color.a);
            fan::print("Found diffuse color:", base_color.r, base_color.g, base_color.b, base_color.a);
          }
        }

        cmaterial->Get(AI_MATKEY_COLOR_AMBIENT, material_data.color[aiTextureType_AMBIENT]);
        cmaterial->Get(AI_MATKEY_COLOR_SPECULAR, material_data.color[aiTextureType_SPECULAR]);
        cmaterial->Get(AI_MATKEY_COLOR_EMISSIVE, material_data.color[aiTextureType_EMISSIVE]);

        return material_data;
      }
      void process_skeleton(aiNode* node, bone_t* parent, fan::vec3& largest_scale) {
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
        if (largest_scale.length() < bone->transformation.get_scale().length()) {
          largest_scale = bone->transformation.get_scale();
        }
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
          process_skeleton(node->mChildren[i], bone, largest_scale);
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
        // corruption when exporting animations, but faster model import
        //scene = (aiScene*)
        importer.ReadFile(path, 
          /*aiProcess_LimitBoneWeights | aiProcess_ImproveCacheLocality |
          aiProcess_RemoveRedundantMaterials | aiProcess_PreTransformVertices |
          aiProcess_FindInvalidData |*/
          aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes | aiProcess_Triangulate | 
          aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | 
          aiProcess_JoinIdenticalVertices | aiProcess_GenBoundingBoxes | aiProcess_SplitLargeMeshes);

        
        scene = importer.GetOrphanedScene();
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
          fan::print(importer.GetErrorString());
          return false;
        }

        //m_transform = scene->mRootNode->mTransformation;
        // for centering
        m_transform = fan::mat4{1};

        bone_count = 0;
        fan::vec3 largest_bone = 0;
        process_skeleton(scene->mRootNode, nullptr, largest_bone);
        load_animations();

        meshes.clear();

        fan::vec3 global_min(FLT_MAX);
        fan::vec3 global_max(-FLT_MAX);

        process_node(scene->mRootNode, aiMatrix4x4(), global_min, global_max);

        aabbmin = global_min;
        aabbmax = global_max;
        
        update_bone_transforms();
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

        segment = std::clamp(segment, uint32_t(0), uint32_t(times.size() - 1));

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
        if (animation.bone_poses.empty()) {
          return;
        }
        // this fmods other animations too. dt per animation? or with weights or max of all animations?
        // fix xd
        if (animation.duration != 0) {
          if (/*animation.weight == 1 && */get_active_animation_id() != -1 && animation.duration == get_active_animation().duration) {
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

        fan::vec3 position = 0, scale = 0;
        fan::quat rotation;
        float total_weight = 0.0f;
        for (const auto& [key, anim] : animation_list) {
          if (anim.weight > 0 && !anim.bone_poses.empty() && !anim.bone_transform_tracks.empty()) {
            const auto& pose = anim.bone_poses[bone.id];
            if (pose.rotation.w == -9999) {
              local_transform = bone.transformation;
            }
            else {
              position += pose.position * anim.weight;
              fan::quat normalized_rotation = pose.rotation.normalized();
              rotation = fan::quat::slerp(rotation, normalized_rotation, anim.weight / (total_weight + anim.weight));
              scale += pose.scale * anim.weight;
              total_weight += anim.weight;
            }
          }
        }

        if (total_weight > 0) {
          rotation = rotation.normalized(); local_transform = fan::translation_matrix(position) * fan::rotation_quat_matrix(rotation) * fan::scaling_matrix(scale);
        }
        else { local_transform = bone.transformation; }

        fan::mat4 global_transform =
          parent_transform *
          local_transform *
          bone.user_transform;

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
          f64_t time_scale = 1000.0 / ticksPerSecond;
          animation.duration = anim->mDuration * time_scale;
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
            fan::model::bone_transform_track_t track;
            for (int j = 0; j < channel->mNumPositionKeys; j++) {
              track.position_timestamps.push_back(channel->mPositionKeys[j].mTime * time_scale);
              track.positions.push_back(channel->mPositionKeys[j].mValue);
            }
            for (int j = 0; j < channel->mNumRotationKeys; j++) {
              track.rotation_timestamps.push_back(channel->mRotationKeys[j].mTime* time_scale);
              track.rotations.push_back(channel->mRotationKeys[j].mValue);
            }
            for (int j = 0; j < channel->mNumScalingKeys; j++) {
              track.scale_timestamps.push_back(channel->mScalingKeys[j].mTime* time_scale);
              track.scales.push_back(channel->mScalingKeys[j].mValue);

            }
            animation.bone_transform_tracks[found->second->id] = track;
          }

        }
      }
      std::string create_an(const std::string& key, f32_t weight = 1.f, f32_t duration = 1.f) {
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
      // works best with fbx
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
          fan::print(std::string("failed to export animation:") + exporter.GetErrorString());
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
        dme_type_t& operator[](uintptr_t i) {
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

        model.iterate_bones(*model.root_bone, [&](fan::model::bone_t& bone) {
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
        model.iterate_bones(*model.root_bone, [&](fan::model::bone_t& bone) {
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
        iterator.iterate_bones(*iterator.root_bone, [&](fan::model::bone_t& bone) {
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
          iterator.iterate_bones(*iterator.root_bone, [&](fan::model::bone_t& bone) {
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
          iterator.iterate_bones(*iterator.root_bone, [&](fan::model::bone_t& bone) {
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
        iterator.iterate_bones(*iterator.root_bone, [&](fan::model::bone_t& bone) {
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
          iterator.iterate_bones(*iterator.root_bone, [&](fan::model::bone_t& bone) {
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
          iterator.iterate_bones(*iterator.root_bone, [&](fan::model::bone_t& bone) {
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
      static fan::mat4 get_world_matrix(fan::model::bone_t* entity, fan::mat4 localMatrix) {
        fan::model::bone_t* parentID = entity->parent;
        while (parentID != nullptr)
        {
          localMatrix = localMatrix * parentID->get_local_matrix();
          parentID = parentID->parent;
        }
        return localMatrix;
      }
      static fan::mat4 get_inverse_parent_matrix(fan::model::bone_t* entity) {
        fan::mat4 inverseParentMatrix(1);
        fan::model::bone_t* parentID = entity->parent;
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
        fan::model::bone_t transform = source_bone;
        transform.position = position;

        fan::mat4 local_matrix = source_bone.world_matrix.inverse() * fan::model::fms_t::get_world_matrix(&source_bone, transform.get_local_matrix());
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

        fan::model::bone_t transform = source_bone;
        transform.rotation = (animation_rotation * tpose_adjust).normalized();

        fan::mat4 local_matrix = source_bone.world_matrix.inverse() * fan::model::fms_t::get_world_matrix(&source_bone, transform.get_local_matrix());
        local_matrix = target_bone.world_matrix * local_matrix * target_bone.inverse_parent_matrix;
        return fan::quat(local_matrix).inverse();
      }
      static fan::vec3 transformScale(
        const fan::vec3& scale,
        bone_t& source_bone,
        bone_t& target_bone
      ) {
        fan::model::bone_t transform = source_bone;
        transform.scale = scale;

        fan::mat4 local_matrix = source_bone.world_matrix.inverse() * fan::model::fms_t::get_world_matrix(&source_bone, transform.get_local_matrix());
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

        anim.iterate_bones(*anim.root_bone, [&](fan::model::bone_t& bone) {
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
        using namespace fan::graphics;

        if (!bone) {
          return;
        }

        gui::tree_node_flags_t flags = gui::tree_node_flags_open_on_arrow | gui::tree_node_flags_framed;
        if (bone->children.empty()) {
          flags |= gui::tree_node_flags_leaf;
        }

        if (depth > 0) {
          gui::indent(10.0f);
        }

        bool node_open = gui::tree_node_ex(bone->name + " | parent " + (bone->parent ? bone->parent->name : "N/A"), flags);

        if (node_open) {
          for (bone_t* child : bone->children) {
            print_bone_recursive(child, depth + 1);
          }
          gui::tree_pop();
        }

        if (depth > 0) {
          gui::unindent(10.0f);
        }
      }

      void print_bones(bone_t* root_bone) {
        using namespace fan::graphics;
        if (!root_bone) {
          gui::text_disabled("No skeleton loaded");
          return;
        }

        gui::begin("Bone Hierarchy");

        if (gui::button("Reset All Rotations")) {
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

        gui::separator();
        print_bone_recursive(root_bone);
        gui::end();
      }

      void display_animations() {
        using namespace fan::graphics;
        for (auto& anim_pair : animation_list) {
          bool nodeOpen = gui::tree_node(anim_pair.first);
          if (nodeOpen) {
            auto& anim = anim_pair.second;

            bool time_stamps_open = gui::tree_node("timestamps");
            if (time_stamps_open) {
              iterate_bones(*root_bone, [&](fan::model::bone_t& bone) {
                auto& bt = anim.bone_transform_tracks[bone.id];
                uint32_t data_count = bt.rotation_timestamps.size();
                if (data_count) {
                  bool node_open = gui::tree_node(bone.name);
                  if (node_open) {
                    for (int i = 0; i < bt.rotation_timestamps.size(); ++i) {
                      gui::drag("rotation:" + std::to_string(i), &bt.rotation_timestamps[i]);
                    }
                    gui::tree_pop();
                  }
                }
              });
              gui::tree_pop();
            }

            bool properties_open = gui::tree_node("properties");
            if (properties_open) {
              iterate_bones(*root_bone, [&](fan::model::bone_t& bone) {
                auto& bt = anim.bone_transform_tracks[bone.id];
                uint32_t data_count = bt.rotations.size();
                if (data_count) {
                  bool node_open = gui::tree_node(bone.name);
                  if (node_open) {
                    for (int i = 0; i < bt.rotations.size(); ++i) {
                      gui::drag("rotation:" + std::to_string(i), bt.rotations[i].data());
                    }
                    gui::tree_pop();
                  }
                }
              });
              gui::tree_pop();
            }

            gui::tree_pop();
          }
        }
      }

      void mouse_modify_joint(f32_t ddt) {
        using namespace fan::graphics;
        static constexpr f64_t delta_time_divier = 1e+6;

        gui::drag("current time", &dt, 1, 0, std::max(longest_animation, 1.f));

        if (gui::checkbox("play animation", &play_animation)) {
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
        if (gui::list_box("animation list", &current_id, animations.data(), animations.size())) {
          active_anim = animations[current_id];
        }

        gui::drag("animation weight", &animation_list[active_anim].weight, 0.01, 0, 1);

        if (active_bone != -1) {
          auto& anim = get_active_animation();
          auto& bt = anim.bone_transform_tracks[active_bone];
          for (int i = 0; i < bt.rotations.size(); ++i) {
            gui::drag("rotations:" + std::to_string(i), bt.rotations[i].data(), 0.01);
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
            gui::text("No bones in current animation");
            return;
          }

          if (gui::button("save keyframe")) {
            fk_set_rot(active_anim, active_bone, current_frame / 1000.f, anim.bone_poses[active_bone].rotation);
          }
        }
      }

      // not very accurate
      bool is_humanoid() const {
        return bone_transforms.size() >= 10;
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
      fan::vec3 aabbmin = 0;
      fan::vec3 aabbmax = 0;
    };
  }
}

#ifdef fms_use_opengl
  #undef fms_use_opengl
#endif

#endif 