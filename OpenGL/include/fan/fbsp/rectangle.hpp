#pragma once

#include <fan/math/vector.hpp>
#include <fan/math/matrix.hpp>

#include <algorithm>
#include <unordered_set>

namespace fan {

	constexpr auto max_stack_amount(100000); // 0.1 mb stack

	static auto grid_2d_to_1d(const fan::vec2& position, f_t view_x) {
		return position.y * view_x + position.x;
	}
	
	fan::mat2ui fbsp_convert_to_src_dst(const fan::vec2& block_size, const fan::vec2& position, const fan::vec2& size, const fan::vec2& velocity = 0) {
		auto first = (position + fan::vec2(velocity.x < 0 ? velocity.x : 0, velocity.y < 0 ? velocity.y : 0)) / block_size;
		auto second = (position + size + fan::vec2(velocity.x > 0 ? velocity.x : 0, velocity.y > 0 ? velocity.y : 0)) / block_size;
		return fan::mat2ui(first, second);
	}

	class base_fbsp {
	public:

		base_fbsp(const fan::vec2& map_size, const fan::vec2& block_size, const std::vector<fan::mat2>& walls) 
			: m_map_size(map_size), m_block_size(block_size), m_map_size_1d(((m_map_size.x / m_block_size.x) * (m_map_size.y / m_block_size.y)) + 1),
			  view(fan::vec2((f_t)m_map_size.x / m_block_size.x, (f_t)m_map_size.y / m_block_size.y).ceiled())  
		{
			m_grid.resize(m_map_size_1d);

			for (uint_t i = 0; i < walls.size(); ++i) {
				const auto& neighbours = this->get_blocks(walls[i][0], walls[i][1]);
				for (int j = 0; j < neighbours.size(); ++j) {
					m_grid[neighbours[j]].emplace_back(i);
				}
			}
		}

		base_fbsp(const fan::vec2& map_size, const fan::vec2& block_size, const std::vector<fan::vec2>& position, const std::vector<fan::vec2>& size) 
			: m_map_size(map_size), m_block_size(block_size), m_map_size_1d(((m_map_size.x / m_block_size.x) * (m_map_size.y / m_block_size.y)) + 1),
			  view(fan::vec2((f_t)m_map_size.x / m_block_size.x, (f_t)m_map_size.y / m_block_size.y).ceiled())  
		{
			m_grid.resize(m_map_size_1d);

			for (uint_t i = 0; i < position.size(); ++i) {
				const auto& neighbours = this->get_blocks(position[i], size[i]);
				for (int j = 0; j < neighbours.size(); ++j) {
					m_grid[neighbours[j]].emplace_back(i);
				}
			}
		}

		base_fbsp(const fan::vec2& map_size, const fan::vec2& block_size, const fan_2d::rectangle_vector& walls) 
			: base_fbsp(map_size, block_size, walls.get_positions(), walls.get_sizes()) { }

		fan::mat2ui convert_to_src_dst(const fan_2d::rectangle& r) const {
			return fbsp_convert_to_src_dst(m_block_size, r.get_position(), r.get_size(), r.get_velocity() * r.m_window.get_delta_time());
		}

		std::vector<uint_t> get_blocks(const fan_2d::rectangle& r) const {
			return get_blocks(r.get_position(), r.get_size(), r.get_velocity() * r.m_window.get_delta_time());
		}

		std::vector<uint_t> get_blocks(const fan::vec2& position, const fan::vec2& size, const fan::vec2& velocity = 0) const {

			const auto& src_dst = fbsp_convert_to_src_dst(m_block_size, position, size, velocity);
			const auto& src = src_dst[0];
			const auto& dst = src_dst[1];

			std::vector<uint_t> indices;

			if (src.x == -1 || src.y == -1 || dst.x == -1 || dst.y == -1) {
				return {};
			}

			for (f_t j = std::min(src.y, dst.y); j <= std::min(src.y, dst.y) + fan::distance(src.y, dst.y); ++j) {
				for (f_t i = std::min(src.x, dst.x); i <= std::min(src.x, dst.x) + fan::distance(src.x, dst.x); ++i) {
					indices.emplace_back(grid_2d_to_1d(fan::vec2(i, j), view.x));
				}
			}

			return indices;
		}

		std::unordered_set<uint_t> get_neighbours(const fan_2d::rectangle& r) {
			return get_neighbours(r.get_position(), r.get_size(), r.get_velocity() * r.m_window.get_delta_time());
		}

		std::unordered_set<uint_t> get_neighbours(const fan::vec2& position, const fan::vec2& size, const fan::vec2& velocity = 0) const {
			const auto& src_dst = fbsp_convert_to_src_dst(m_block_size, position, size, velocity);
			const fan::vec2ui& src = src_dst[0];
			const fan::vec2ui& dst = src_dst[1];

			std::unordered_set<uint_t> walls;

			for (f_t j = std::min(src.y, dst.y); j <= std::min(src.y, dst.y) + fan::distance(src.y, dst.y); ++j) {
				for (f_t i = std::min(src.x, dst.x); i <= std::min(src.x, dst.x) + fan::distance(src.x, dst.x); ++i) {
					const uint_t id = grid_2d_to_1d(fan::vec2(i, j), view.x);
					walls.insert(m_grid[id].begin(), m_grid[id].end());
				}
			}

			return walls;
		}

	private:

		fan::vec2 m_map_size;
		fan::vec2 m_block_size;
		uint_t m_map_size_1d;
		fan::vec2 view;

		std::vector<std::vector<uint_t>> m_grid;

	};

	using fbsp = fan::base_fbsp;
}