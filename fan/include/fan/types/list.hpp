#pragma once

#include <fan/types/vector.hpp>

namespace fan {

	template <typename type_t, std::size_t Rows, std::size_t Cols>
	struct _da;

	template <typename type_t, std::size_t Rows>
	struct list;

	template <typename type_t, std::size_t Rows, std::size_t Cols = 1>
	using da_t = std::conditional_t<Cols == 1, list<type_t, Rows>, _da<type_t, Rows, Cols>>;

	template <typename type_t, std::size_t rows>
	struct list : public std::array<type_t, rows> {

		using array_type = std::array<type_t, rows>;

		template <typename T>
		constexpr list(const _vec2<T>& vector) {
			for (uintptr_t i = 0; i < std::min(rows, vector.size()); i++) {
				this->operator[](i) = vector[i];
			}
		}

		template <typename T>
		constexpr list(const _vec3<T>& vector) {
			for (uintptr_t i = 0; i < std::min(rows, vector.size()); i++) {
				this->operator[](i) = vector[i];
			}
		}

		template <typename T>
		constexpr list(const _vec4<T>& vector) {
			for (uintptr_t i = 0; i < std::min(rows, vector.size()); i++) {
				this->operator[](i) = vector[i];
			}
		}

		template <typename ...T>
		constexpr list(T... x) : std::array<type_t, rows>{ (type_t)x... } {}

		template <typename T>
		constexpr list(T value) : std::array<type_t, rows>{0} {
			std::fill(this->begin(), this->end(), value);
		}

		template <typename T, std::size_t array_n>
		constexpr list(const list<T, array_n>& list) {
			std::copy(list.cbegin(), list.cend(), this->begin());
		}

		constexpr auto operator++() noexcept {
			return this->data() + 1;
		}

		template <typename T, std::size_t list_n>
		constexpr list operator+(const list<T, list_n>& _list) const noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			list calculation_list;
			for (uintptr_t i = 0; i < rows; i++) {
				calculation_list[i] = this->operator[](i) + _list[i];
			}
			return calculation_list;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator+(T value) const noexcept {
			list list;
			for (uintptr_t i = 0; i < rows; i++) {
				list[i] = this->operator[](i) + value;
			}
			return list;
		}

		template <typename T, std::size_t rows_>
		constexpr list operator+=(const list<T, rows_>& value) noexcept {
			//static_assert(rows >= list_n, "second list is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) += value[i];
			}
			return *this;
		}


		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator+=(T value) noexcept {
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) += value;
			}
			return *this;
		}

		constexpr list operator-() const noexcept {
			list l;
			for (uintptr_t i = 0; i < rows; i++) {
				l[i] = -this->operator[](i);
			}
			return l;
		}

		template <typename T, std::size_t list_n>
		constexpr list operator-(const list<T, list_n>& _list) const noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			list calculation_list;
			for (uintptr_t i = 0; i < rows; i++) {
				calculation_list[i] = this->operator[](i) - _list[i];
			}
			return calculation_list;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator-(T value) const noexcept {
			list list;
			for (uintptr_t i = 0; i < rows; i++) {
				list[i] = this->operator[](i) - value;
			}
			return list;
		}

		template <typename T, std::size_t list_n>
		constexpr list operator-=(const list<T, list_n>& value) noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) -= value[i];
			}
			return *this;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator-=(T value) noexcept {
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) -= value;
			}
			return *this;
		}

		template <typename T, std::size_t list_n>
		constexpr list operator*(const list<T, list_n>& _list) const noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			list calculation_list;
			for (uintptr_t i = 0; i < rows; i++) {
				calculation_list[i] = this->operator[](i) * _list[i];
			}
			return calculation_list;
		}

		constexpr list operator*(type_t value) const noexcept {
			list list;
			for (uintptr_t i = 0; i < rows; i++) {
				list[i] = this->operator[](i) * value;
			}
			return list;
		}

		template <typename T, std::size_t list_n>
		constexpr list operator*=(const list<T, list_n>& value) noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) *= value[i];
			}
			return *this;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator*=(T value) noexcept {
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) *= value;
			}
			return *this;
		}


		template <typename T, std::size_t list_n>
		constexpr list operator/(const list<T, list_n>& _list) const noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			list calculation_list;
			for (uintptr_t i = 0; i < rows; i++) {
				calculation_list[i] = this->operator[](i) / _list[i];
			}
			return calculation_list;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator/(T value) const noexcept {
			list list;
			for (uintptr_t i = 0; i < rows; i++) {
				list[i] = this->operator[](i) / value;
			}
			return list;
		}

		template <typename T, std::size_t list_n>
		constexpr list operator/=(const list<T, list_n>& value) noexcept {
			static_assert(rows >= list_n, "second list is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) /= value[i];
			}
			return *this;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr list operator/=(T value) noexcept {
			for (uintptr_t i = 0; i < rows; i++) {
				this->operator[](i) /= value;
			}
			return *this;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr auto operator%(T value) {
			list l;
			for (uintptr_t i = 0; i < rows; i++) {
				l = fmodf(this->operator[](i), value);
			}
			return l;
		}

		constexpr bool operator<(const list<type_t, rows>& list_) {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i) < list_[i]) {
					return true;
				}
			}
			return false;
		}

		constexpr bool operator<=(const list<type_t, rows>& list_) {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i) <= list_[i]) {
					return true;
				}
			}
			return false;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr bool operator==(T value) {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i) != value) {
					return false;
				}
			}
			return true;
		}

		constexpr bool operator==(const list<type_t, rows>& list_) {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i) != list_[i]) {
					return false;
				}
			}
			return true;
		}

		template <typename T, typename = std::enable_if_t<!std::is_same_v<T, da_t<type_t, rows>>>>
		constexpr bool operator!=(T value) {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i) == value) {
					return false;
				}
			}
			return true;
		}

		template <typename T>
		constexpr bool operator!=(const list<T, rows>& list_) {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i) == list_[i]) {
					return false;
				}
			}
			return true;
		}

		constexpr auto& operator*() {
			return *this->begin();
		}

		constexpr void print() const {
			for (uintptr_t i = 0; i < rows; i++) {
				std::cout << this->operator[](i) << ((i + 1 != rows) ? " " : "\rows");
			}
		}

		constexpr auto u() const noexcept {
			return this->operator-();
		}

		constexpr auto min() const noexcept {
			return *std::min_element(this->begin(), this->end());
		}

		constexpr auto max() const noexcept {
			return *std::max_element(this->begin(), this->end());
		}

		constexpr auto avg() const noexcept {
			return std::accumulate(this->begin(), this->end(), 0) / this->size();
		}

		constexpr auto abs() const noexcept {
			list l;
			for (uintptr_t i = 0; i < rows; i++) {
				l[i] = std::abs(this->operator[](i));
			}
			return l;
		}

		constexpr list<type_t, 2> floor() const noexcept {
			list l;
			for (uintptr_t i = 0; i < rows; i++) {
				l[i] = std::floor(this->operator[](i));
			}
			return l;
		}

		constexpr list<type_t, 2> ceil() const noexcept {
			list l;
			for (uintptr_t i = 0; i < rows; i++) {
				l[i] = (this->operator[](i) < 0 ? -std::ceil(-this->operator[](i)) : std::ceil(this->operator[](i)));
			}
			return l;
		}

		constexpr list<type_t, 2> round() const noexcept {
			list l;
			for (uintptr_t i = 0; i < rows; i++) {
				l[i] = std::round(-this->operator[](i));
			}
			return l;
		}

		constexpr type_t pmax() const noexcept {
			decltype(*this) list = this->abs();
			auto biggest = std::max_element(list.begin(), list.end());
			if (this->operator[](biggest - list.begin()) < 0) {
				return -*biggest;
			}
			return *biggest;
		}

		constexpr auto gfne() const noexcept {
			for (uintptr_t i = 0; i < rows; i++) {
				if (this->operator[](i)) {
					return this->operator[](i);
				}
			}
			return type_t();
		}
	};

}