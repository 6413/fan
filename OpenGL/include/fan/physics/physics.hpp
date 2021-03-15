#pragma once

#include <fan/types/types.hpp>


namespace fan_2d {

	namespace physics_2d {

		class physics {
		public:
			
			physics() : m_mass(0), m_friction(0) { }

			physics(f_t mass, f_t friction) : m_mass(0), m_friction(0) { }

			f_t get_mass() const {
				return m_mass;
			}
			void set_mass(f_t mass) {
				m_mass = mass;
			}

			f_t get_friction() const {
				return m_friction;
			}
			void set_friction(f_t friction) {
				m_friction = friction;
			}

		protected:

			f_t m_mass;
			f_t m_friction;

		};

	}

}