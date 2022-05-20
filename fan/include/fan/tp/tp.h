#pragma once

#include <memory>

// reference https://web.archive.org/web/20170703203916/http://clb.demon.fi/projects/rectangle-bin-packing

namespace fan {
  namespace tp {

    struct texture_pack {
			struct texture {
				std::shared_ptr<texture> left;
				std::shared_ptr<texture> right;

				// The top-left coordinate of the rectangle.
				fan::vec2i position;
				fan::vec2i size;

				friend std::ostream& operator<<(std::ostream& os, const texture& tex) {
					os << '{' << "\n position:" << tex.position << "\n size:" << tex.size << "\n}";
					return os;
				}
			};

			void open(const fan::vec2& size) {
				bin_size = size;
				root.left = root.right = 0;
				root.position = 0;
				root.size = size;
			}
			void close() {

			}

			texture *push(const fan::vec2& size) {
				return push(&root, size);
			}

			f32_t occupancy() const {
				uint32_t totalSurfaceArea = bin_size.x + bin_size.y;
				uint32_t usedSurfaceArea = used_surface_area(root);

				return (f32_t)usedSurfaceArea/totalSurfaceArea;
			}

		private:
			texture root;

			fan::vec2i bin_size;

			unsigned long used_surface_area(const texture &node) const {
				if (node.left || node.right) {
					unsigned long usedSurfaceArea = node.size.x * node.size.y;
					if (node.left) {
						usedSurfaceArea += used_surface_area(*node.left);
					}
					if (node.right) {
						usedSurfaceArea += used_surface_area(*node.right);
					}
					return usedSurfaceArea;
				}
				return 0;
			}

			texture *push(texture *node, const fan::vec2i& size) {
				if (node->left || node->right) {
					if (node->left) {
						texture *newNode = push(node->left.get(), size);
						if (newNode)
							return newNode;
					}
					if (node->right) {
						texture *newNode = push(node->right.get(), size);
						if (newNode)
							return newNode;
					}
					return nullptr;
				}

				if (size.x > node->size.x || size.y > node->size.y) {
					return nullptr;
				}

				int w = node->size.x - size.x;
				int h = node->size.y - size.y;
				node->left = std::make_shared<texture>();
				node->right = std::make_shared<texture>();
				if (w <= h) {
					node->left->position.x = node->position.x + size.x;
					node->left->position.y = node->position.y;
					node->left->size.x = w;
					node->left->size.y = size.y;

					node->right->position.x = node->position.x;
					node->right->position.y = node->position.y + size.y;
					node->right->size.x = node->size.x;
					node->right->size.y = h;
				}
				else {
					node->left->position.x = node->position.x;
					node->left->position.y = node->position.y + size.y;
					node->left->size.x = size.x;
					node->left->size.y = h;

					node->right->position.x = node->position.x + size.x;
					node->right->position.y = node->position.y;
					node->right->size.x = w;
					node->right->size.y = node->size.y;
				}
				node->size = size;
				return node;
			}

		};
  }
}
