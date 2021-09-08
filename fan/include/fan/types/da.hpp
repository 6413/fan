#pragma once

#include <array>
#include <algorithm>
#include <iostream>
#include <numeric>

#include <fan/types/list.hpp>

namespace fan {

	template <typename type_t, std::size_t Rows, std::size_t Cols>
	struct _da {

		list<type_t, Cols> m[Rows];

		using matrix_type = _da<type_t, Rows, Cols>;
		using value_type = list<type_t, Cols>;

		static constexpr std::size_t rows = Rows;
		static constexpr std::size_t cols = Cols;

		constexpr _da() : m{ 0 } { }

		template <typename T, typename ..._Type, typename = std::enable_if_t<!std::is_arithmetic_v<T>>>
		constexpr _da(T first, _Type... value) {
		//	static_assert(sizeof...(value) >= rows, "more elements than dat's size");
			int init = 0;
			((value_type*)m)[init++] = first;
			((((value_type*)m)[init++] = value), ...);
		}

		// ignores other values like vector
		template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
		constexpr _da(T value) : m{ 0 } {
			for (uintptr_t i = 0; i < rows && i < cols; i++) {
				m[i][i] = value;
			}
		}

	#ifdef ASSIMP_API
		constexpr _da (const aiMatrix4x4& matrix) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					this->m[i][j] = matrix[j][i];
				}
			}
		}
	#endif

		template <typename T, std::size_t Rows2, std::size_t Cols2>
		constexpr matrix_type operator+(const _da<T, Rows2, Cols2>& matrix) const noexcept { // matrix
			matrix_type _matrix;
			static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] + matrix[i];
			}
			return _matrix;
		}

		template <typename T, std::size_t da_t_n>
		constexpr matrix_type operator+(const list<T, da_t_n>& da_t) const noexcept { // list
			matrix_type _matrix;
			static_assert(cols >= da_t_n, "list is bigger than the matrice's Rows");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] + da_t;
			}
			return _matrix;
		}

		template <typename T>
		constexpr matrix_type operator+(T value) const noexcept { // basic value
			matrix_type _matrix;
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] + value;
			}
			return _matrix;
		}

		template <typename T, std::size_t Rows2, std::size_t Cols2>
		constexpr auto operator+=(const _da<T, Rows2, Cols2>& matrix) noexcept { // matrix
			matrix_type _matrix;
			static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] += matrix[i];
			}
			return _matrix;
		}

		template <typename T, std::size_t da_t_n>
		constexpr matrix_type operator+=(const list<T, da_t_n>& da_t) noexcept { // list
			matrix_type _matrix;
			static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] += da_t;
			}
			return _matrix;
		}

		template <typename T>
		constexpr matrix_type operator+=(T value) noexcept { // basic value
			matrix_type _matrix;
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] += value;
			}
			return _matrix;
		}

		template <typename T, std::size_t Rows2, std::size_t Cols2>
		constexpr matrix_type operator-(const _da<T, Rows2, Cols2>& matrix) const noexcept { // matrix
			matrix_type _matrix;
			static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] - matrix[i];
			}
			return _matrix;
		}

		template <typename T, std::size_t da_t_n>
		constexpr matrix_type operator-(const list<T, da_t_n>& da_t) const noexcept { // list
			matrix_type _matrix;
			static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] - da_t;
			}
			return _matrix;
		}

		template <typename T>
		constexpr matrix_type operator-(T value) const noexcept { // basic value
			matrix_type _matrix;
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] - value;
			}
			return _matrix;
		}

		template <typename T, std::size_t Rows2, std::size_t Cols2>
		constexpr matrix_type operator-=(const _da<T, Rows2, Cols2>& matrix) noexcept { // matrix
			matrix_type _matrix;
			static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] -= matrix[i];
			}
			return _matrix;
		}

		template <typename T, std::size_t da_t_n>
		constexpr matrix_type operator-=(const list<T, da_t_n>& da_t) noexcept { // list
			matrix_type _matrix;
			static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] -= da_t;
			}
			return _matrix;
		}

		template <typename T>
		constexpr matrix_type operator-=(T value) noexcept { // basic value
			matrix_type _matrix;
			for (uintptr_t i = 0; i < rows; i++) {
				_matrix[i] = m[i] -= value;
			}
			return _matrix;
		}

		constexpr matrix_type operator-() noexcept {
			for (int _I = 0; _I < Rows; _I++) {
				for (int _J = 0; _J < Cols; _J++) {
					m[_I][_J] = -m[_I][_J];
				}
			}
			return *this;
		}

		constexpr matrix_type operator*(const _da<type_t, Rows, Cols>& _Lhs) noexcept(false) {
			if (Rows != Cols) {
				throw("first matrix rows must be same as second's colums");
			}
			auto SrcA0 = m[0];
			auto SrcA1 = m[1];
			auto SrcA2 = m[2];
			auto SrcA3 = m[3];
			
			auto SrcB0 = _Lhs[0];
			auto SrcB1 = _Lhs[1];
			auto SrcB2 = _Lhs[2];
			auto SrcB3 = _Lhs[3];

			matrix_type Result;
			Result[0] = SrcA0 * SrcB0[0] + SrcA1 * SrcB0[1] + SrcA2 * SrcB0[2] + SrcA3 * SrcB0[3];
			Result[1] = SrcA0 * SrcB1[0] + SrcA1 * SrcB1[1] + SrcA2 * SrcB1[2] + SrcA3 * SrcB1[3];
			Result[2] = SrcA0 * SrcB2[0] + SrcA1 * SrcB2[1] + SrcA2 * SrcB2[2] + SrcA3 * SrcB2[3];
			Result[3] = SrcA0 * SrcB3[0] + SrcA1 * SrcB3[1] + SrcA2 * SrcB3[2] + SrcA3 * SrcB3[3];
			return Result;
		}

		template <typename T>
		constexpr auto operator*=(T value) {
			return this->operator[]<true>(0) *= value;
		}

		template <typename T>
		constexpr auto operator/(T value) const noexcept {
			return this->operator[]<true>(0) / value;
		}

		template <typename T>
		constexpr auto operator/=(T value) {
			return this->operator[]<true>(0) /= value;
		}

		template <typename T>
		constexpr auto operator%(T value) {
			for (uintptr_t i = 0; i < Cols; i++) {
				for (uintptr_t j = 0; j < Rows; j++) {
					m[i][j] = fmodf(m[i][j], value);
				}
			}
			return *this;
		}

		constexpr bool operator==(const _da<type_t, rows, cols>& list_) {
			for (uintptr_t i = 0; i < rows; i++) {
				for (uintptr_t j = 0; j < cols; j++) {
					if (m[i][j] != list_[i][j]) {
						return false;
					}
				}
			}
			return true;
		}

		template <bool return_array = false>
		constexpr auto operator[](std::size_t i) const {
			return m[i];
		}

		template <bool return_array = false>
		constexpr auto& operator[](std::size_t i) {
			return m[i];
		}

		constexpr void print() const {
			for (uintptr_t i = 0; i < rows; i++) {
				m[i].print();
			}
		}

		list<type_t, Cols>* begin() noexcept {
			return &m[0];
		}

		list<type_t, Cols>* end() noexcept {
			return &m[rows];
		}

		constexpr auto data() noexcept {
			return &m[0][0];
		}

		constexpr auto u() noexcept {
			return this->operator-();
		}

		constexpr auto min() noexcept {
			return *std::min_element(begin(), end());
		}

		constexpr auto max() noexcept {
			return *std::max_element(begin(), end());
		}

		constexpr auto avg() noexcept {
			list<type_t, cols> averages;
			for (uintptr_t i = 0; i < cols; i++) {
				averages += m[i];
			}
			return averages / cols;
		}


		constexpr auto vector() noexcept {
			return std::vector<da_t<type_t, cols>>(this->begin(), this->end());
		}

		constexpr auto size() const noexcept {
			return rows;
		}
	};

	//template <typename T>
	//constexpr bool dcom_fr(uintptr_t n, T x, T y) noexcept {
	//	switch (n) {
	//	case 0: {
	//		return x < y;
	//	}
	//	case 1: {
	//		return x > y;
	//	}
	//	}
	//	return false;
	//}

	//using mat2x2 = _da<f32_t, 2, 2>;
	using mat2x3 = _da<f32_t, 2, 3>;
	using mat3x2 = _da<f32_t, 3, 2>;
	using mat4x2 = _da<f32_t, 4, 2>;
	using mat3x3 = _da<f32_t, 3, 3>;

	//using mat2 = mat2x2;
	using mat3 = mat3x3;

	using da2 = da_t<f32_t, 2>;
	using da3 = da_t<f32_t, 3>;

	template <typename T, std::size_t rows>
	std::ostream& operator<<(std::ostream& os, const list<T, rows> list_) noexcept
	{
		for (uintptr_t i = 0; i < rows; i++) {
			os << list_[i] << ' ';
		}
		return os;
	}

	template <
		template <typename, std::size_t, std::size_t> typename da_t_t,
		typename T, std::size_t rows, std::size_t cols
	>
		std::ostream& operator<<(std::ostream& os, const da_t_t<T, rows, cols>& da_t_) noexcept
	{
		for (uintptr_t i = 0; i < rows; i++) {
			for (uintptr_t j = 0; j < cols; j++) {
				os << da_t_[i][j] << ' ';
			}
			os << '\n';
		}
		return os;
	}

	
	template <typename T, typename T2>
	static void copy_matrix4x4(T& dst, const T2& src) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				dst[i][j] = src[i][j];
			}
		}
	}

	template <typename T, typename T2>
	static void copy_matrix4x4_arr(T& dst, const T2& src) {
		int k = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				dst[i][j] = src[k++];
			}
		}
	}

}