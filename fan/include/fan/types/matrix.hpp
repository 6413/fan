#pragma once

#include <fan/types/vector.hpp>
#include <fan/types/quaternion.hpp>

#include <type_traits>

#include <iostream>
#include <exception>

namespace fan {

	template<std::size_t N>
	struct matrix_helper {
		template <typename T, fan::is_arithmetic_t T2>
		static constexpr void construct_diagonal(T& matrix, T2 value)
		{
			matrix[N-1][N-1] = value;
			matrix_helper<N-1>::construct_diagonal(matrix, value);
		}
	};

	template <> 
	struct matrix_helper<0> {
		template <typename T, typename T2>
		static constexpr void construct_diagonal(T& arr, T2 value) {}
	};

	template <typename _Ty, std::size_t x, std::size_t y, std::size_t original_value>
	struct matrix_operators {

		template <fan::is_arithmetic_t... T>
		static constexpr _Ty construct(_Ty& matrix, const T... lhs)
		{
			matrix[y][x] = std::get<y * (original_value + 1) + x>(std::forward_as_tuple(lhs...));

			if constexpr (!x) {
				matrix_operators<_Ty, original_value, y - 1, original_value>::construct(matrix, lhs...);
			}
			else {
				matrix_operators<_Ty, x - 1, y, original_value>::construct(matrix, lhs...);
			}

			return matrix;
		}

		template <typename T>
		static constexpr _Ty addition(_Ty& matrix, const _Ty& lhs, const T& rhs)
		{
			matrix[x][y] = lhs[x][y] + rhs[x][y];

			if constexpr (!y) {
				matrix_operators<_Ty, x - 1, original_value, original_value>::addition(matrix, lhs, rhs);
			}
			else {
				matrix_operators<_Ty, x, y - 1, original_value>::addition(matrix, lhs, rhs);
			}

			return matrix;
		}

		template <fan::is_arithmetic_t T>
		static constexpr _Ty addition(_Ty& matrix, const _Ty& lhs, const T& rhs)
		{
			matrix[x][y] = lhs[x][y] + rhs;

			if constexpr (y <= 0) {
				matrix_operators<_Ty, x - 1, original_value, original_value>::addition(matrix, lhs, rhs);
			}
			else {
				matrix_operators<_Ty, x, y - 1, original_value>::addition(matrix, lhs, rhs);
			}
			return matrix;
		}

		template <typename T>
		static constexpr _Ty substraction(_Ty& matrix, const _Ty& lhs, const T& rhs)
		{
			matrix[x][y] = lhs[x][y] - rhs[x][y];

			if constexpr (y <= 0) {
				matrix_operators<_Ty, x - 1, original_value, original_value>::substraction(matrix, lhs, rhs);
			}
			else {
				matrix_operators<_Ty, x, y - 1, original_value>::substraction(matrix, lhs, rhs);
			}
			return matrix;
		}

		static constexpr _Ty negation(_Ty& matrix)
		{
			matrix[x][y] = -matrix[x][y];

			if constexpr (y <= 0) {
				matrix_operators<_Ty, x - 1, original_value, original_value>::negation(matrix);
			}
			else {
				matrix_operators<_Ty, x, y - 1, original_value>::negation(matrix);
			}
			return matrix;
		}

	private:

		template <typename T, std::size_t i>
		static constexpr _Ty multiplication_helper(_Ty& matrix, const _Ty& lhs, const T& rhs) {

			matrix[y][x] += lhs[i][x] * rhs[y][i];

			if constexpr (i > 0) {
				matrix_operators<_Ty, x, y, original_value>::multiplication_helper<T, i - 1>(matrix, lhs, rhs);
			}

			return matrix;
		}

	public:

		template <typename T>
		static constexpr _Ty multiplication(_Ty& matrix, const _Ty& lhs, const T& rhs)
		{

			matrix_operators<_Ty, x, y, original_value>::multiplication_helper<T, original_value>(matrix, lhs, rhs);

			if constexpr (y <= 0) {
				matrix_operators<_Ty, x - 1, original_value, original_value>::multiplication(matrix, lhs, rhs);
			}
			else if (x >= 0) {
				matrix_operators<_Ty, x, y - 1, original_value>::multiplication(matrix, lhs, rhs);
			}

			return matrix;
		}

	};

	template <typename _Ty, std::size_t original_value>
	struct matrix_operators<_Ty, 0, 0, original_value>  {

		template <fan::is_arithmetic_t... T>
		static constexpr _Ty construct(_Ty& matrix, const T... lhs)
		{
			matrix[0][0] = std::get<0>(std::forward_as_tuple(lhs...));

			return matrix;
		}

		template <typename T>
		static constexpr _Ty addition(_Ty& matrix, const _Ty& lhs, const T& rhs) {
			matrix[0][0] = lhs[0][0] + rhs[0][0];

			return matrix;
		}

		template <fan::is_arithmetic_t T>
		static constexpr _Ty addition(_Ty& matrix, const _Ty& lhs, const T& rhs)
		{
			matrix[0][0] = lhs[0][0] + rhs;

			return matrix;
		}

		template <typename T>
		static constexpr _Ty substraction(_Ty& matrix, const _Ty& lhs, const T& rhs) {
			matrix[0][0] = lhs[0][0] - rhs[0][0];

			return matrix;
		}

		static constexpr _Ty negation(_Ty& matrix) {

			matrix[0][0] = -matrix[0][0];

			return matrix;
		}

	private:

		template <typename T, std::size_t i>
		static constexpr _Ty multiplication_helper(_Ty& matrix, const _Ty& lhs, const T& rhs) {

			matrix[0][0] += lhs[i][0] * rhs[0][i];

			if constexpr (i > 0) {
				multiplication_helper<T, i - 1>(matrix, lhs, rhs);
			}

			return matrix;
		}

	public:

		template <typename T>
		static constexpr _Ty multiplication(_Ty& matrix, const _Ty& lhs, const T& rhs) {

			matrix_operators::multiplication_helper<T, original_value>(matrix, lhs, rhs);

			return matrix;
		}
	};

	template <typename type_t, std::size_t size_x, std::size_t size_y>
	struct basic_matrix : public std::array<std::array<type_t, size_y>, size_x> {
	public:

		template <typename T>
		using inherited_type_t = std::array<std::array<T, size_y>, size_x>;

		template <typename T>
		using basic_matrix_t = basic_matrix<T, size_x, size_y>;

		constexpr basic_matrix() : inherited_type_t<type_t>{} {}

		template <fan::is_arithmetic_t T>
		constexpr basic_matrix(T value) : inherited_type_t<type_t>{} 
		{
			matrix_helper<size_x>::construct_diagonal(*this, value);
		}

		template <fan::is_arithmetic_t ...T>
		constexpr basic_matrix(const T&&... value)
		{

			static_assert(!(sizeof...(value) < size_x * size_y), "too few arguments"); // clang can't use size() even though constexpr gg
			static_assert(!(sizeof...(value) > size_x * size_y), "too many arguments"); // clang can't use size() even though constexpr gg

			matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_x - 1>::construct(*this, value...);

		}

		template <fan::is_arithmetic_t T>
		constexpr basic_matrix_t<type_t> operator+(T value) const 
		{
			basic_matrix_t<type_t> matrix_;
			return matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_y - 1>::addition(matrix_, *this, value);
		}

		template <typename T>
		constexpr basic_matrix_t<type_t> operator+(const basic_matrix_t<T>& matrix) const 
		{
			basic_matrix_t<type_t> matrix_;
			return matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_y - 1>::addition(matrix_, *this, matrix);
		}

		template <fan::is_arithmetic_t T>
		constexpr basic_matrix_t<type_t>& operator+=(T value) 
		{
			return *this = this->operator+(value);
		}

		template <typename T>
		constexpr basic_matrix_t<type_t>& operator+=(const basic_matrix_t<T>& matrix) 
		{
			return *this = this->operator+(matrix);
		}

		constexpr basic_matrix_t<type_t> operator-() const
		{
			basic_matrix_t<type_t> matrix(*this);
			return matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_y - 1>::negation(matrix);
		}

		template <typename T>
		constexpr basic_matrix_t<type_t> operator-(const basic_matrix_t<T>& matrix) const 
		{
			basic_matrix_t<type_t> matrix_;
			return matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_y - 1>::substraction(matrix_, *this, matrix);
		}

		template <typename T>
		constexpr basic_matrix_t<type_t>& operator-=(const basic_matrix_t<T>& matrix) 
		{
			return *this = this->operator-(matrix);
		}

		template <typename T>
		constexpr basic_matrix_t<type_t> operator*(const basic_matrix_t<T>& matrix) const 
		{
			basic_matrix_t<type_t> matrix_;

			return matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_y - 1>::multiplication(matrix_, *this, matrix);
		}

		template <typename T>
		constexpr basic_matrix_t<type_t> operator*=(const basic_matrix_t<T>& matrix)
		{
			basic_matrix_t<type_t> matrix_;

			return *this = matrix_operators<basic_matrix_t<type_t>, size_x - 1, size_y - 1, size_y - 1>::multiplication(matrix_, *this, matrix);
		}

		static constexpr uintptr_t size() 
		{
			return size_x * size_y;
		}
	};

		template <typename T, std::size_t x, std::size_t y>
		std::ostream& operator<<(std::ostream& os, const basic_matrix<T, x, y>& matrix)
	{
		for (uintptr_t i = 0; i < x; i++) {
			for (uintptr_t j = 0; j < y; j++) {
				os << matrix[j][i] << ' ';
			}
			os << '\n';
		}
		return os;
	}

	template <typename type_t> 
	class _matrix2x2 : public basic_matrix<type_t, 2, 2> {
	public:

		using matrix_t = basic_matrix<type_t, 2, 2>;

		using basic_matrix<type_t, 2, 2>::basic_matrix;

		template <fan::is_arithmetic_t T>
		constexpr _matrix2x2(T x, T y) : _matrix2x2(fan::_vec2<T>(x), fan::_vec2<T>(y)) {}

		template <typename T, typename T2>
		constexpr _matrix2x2(const fan::_vec2<T>& v, const fan::_vec2<T2>& v2) : basic_matrix<type_t, 2, 2>((type_t)v.x, (type_t)v.y, (type_t)v2.x, (type_t)v2.y) {}

		template <typename T, typename T2>
		constexpr _matrix2x2(fan::_vec2<T>&& v, fan::_vec2<T2>&& v2) : basic_matrix<type_t, 2, 2>(std::move(v.x), std::move(v.y), std::move(v2.x), std::move(v2.y)) {}

		template <typename T>
		constexpr _matrix2x2 operator+(const fan::_vec2<T>& v) const {
			return _matrix2x2((*this)[0][0] + v[0], (*this)[0][1] + v[1], (*this)[1][0] + v[0], (*this)[1][1] + v[1]);
		}

		constexpr decltype(auto) operator[](const uintptr_t i) const {
			switch (i) {
				case 0:
				{
					return fan::_vec2<type_t>(matrix_t::operator[](0).operator[](0), matrix_t::operator[](0).operator[](1));
				}
				case 1:
				{
					return fan::_vec2<type_t>(matrix_t::operator[](1).operator[](0), matrix_t::operator[](1).operator[](1));
				}
				default:
				{
					throw std::runtime_error("out of range");
					return fan::_vec2<type_t>();
				}
			}
		}

		constexpr decltype(auto) operator[](const uintptr_t i) {
			switch (i) {
				case 0:
				{
					return fan::_vec2<type_t>(matrix_t::operator[](0).operator[](0), matrix_t::operator[](0).operator[](1));
				}
				case 1:
				{
					return fan::_vec2<type_t>(matrix_t::operator[](1).operator[](0), matrix_t::operator[](1).operator[](1));
				}
				default:
				{
					throw std::runtime_error("out of range");
					return fan::_vec2<type_t>();
				}
			}
		}

	private:


	};

	template <typename type_t> 
	class _matrix4x4 : public basic_matrix<type_t, 4, 4>{
	public:

		using basic_matrix<type_t, 4, 4>::basic_matrix;

		template <typename T>
		constexpr _matrix4x4(const basic_matrix<T, 4, 4>& matrix) : basic_matrix<type_t, 4, 4>(matrix) {}

		template <typename T>
		constexpr _matrix4x4(const fan::quaternion<T>& quat) : _matrix4x4<type_t>(1) {
			f32_t qxx(quat[0] * quat[0]);
			f32_t qyy(quat[1] * quat[1]);
			f32_t qzz(quat[2] * quat[2]);
			f32_t qxz(quat[0] * quat[2]);
			f32_t qxy(quat[0] * quat[1]);
			f32_t qyz(quat[1] * quat[2]);
			f32_t qwx(quat[3] * quat[0]);
			f32_t qwy(quat[3] * quat[1]);
			f32_t qwz(quat[3] * quat[2]);

			(*this)[0][0] = f32_t(1) - f32_t(2) * (qyy +  qzz);
			(*this)[0][1] = f32_t(2) * (qxy + qwz);
			(*this)[0][2] = f32_t(2) * (qxz - qwy);

			(*this)[1][0] = f32_t(2) * (qxy - qwz);
			(*this)[1][1] = f32_t(1) - f32_t(2) * (qxx +  qzz);
			(*this)[1][2] = f32_t(2) * (qyz + qwx);
			
			(*this)[2][0] = f32_t(2) * (qxz + qwy);
			(*this)[2][1] = f32_t(2) * (qyz - qwx);
			(*this)[2][2] = f32_t(1) - f32_t(2) * (qxx +  qyy);
		}

		
	#ifdef ASSIMP_API
		constexpr _matrix4x4 (const aiMatrix4x4& matrix) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					(*this)[i][j] = matrix[j][i];
				}
			}
		}
	#endif

		constexpr fan::vec3 get_translation() const {
			return fan::vec3((*this)[3][0], (*this)[3][1], (*this)[3][2]);
		}

		constexpr _matrix4x4 translate(const fan::vec3& v) const {
			_matrix4x4 matrix((*this));
			matrix[3][0] = (*this)[0][0] * v[0] + (*this)[1][0] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][0] * v[2])) + (*this)[3][0];
			matrix[3][1] = (*this)[0][1] * v[0] + (*this)[1][1] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][1] * v[2])) + (*this)[3][1];
			matrix[3][2] = (*this)[0][2] * v[0] + (*this)[1][2] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][2] * v[2])) + (*this)[3][2];
			matrix[3][3] = (*this)[0][3] * v[0] + (*this)[1][3] * v[1] + (v.size() < 3 ? + 0 : ((*this)[2][3] * v[2])) + (*this)[3][3];
			return matrix;
		}

		constexpr fan::vec3 get_scale() const {
			return fan::vec3((*this)[0][0], (*this)[1][1], (*this)[2][2]);
		}

		constexpr _matrix4x4 scale(const fan::vec3& v) const {
			_matrix4x4 matrix{};

			matrix[0][0] = (*this)[0][0] * v[0];
			matrix[0][1] = (*this)[0][1] * v[0];
			matrix[0][2] = (*this)[0][2] * v[0];

			matrix[1][0] = (*this)[1][0] * v[1];
			matrix[1][1] = (*this)[1][1] * v[1];
			matrix[1][2] = (*this)[1][2] * v[1];

			matrix[2][0] = (v.size() < 3 ? 0 : (*this)[2][0] * v[2]);
			matrix[2][1] = (v.size() < 3 ? 0 : (*this)[2][1] * v[2]);
			matrix[2][2] = (v.size() < 3 ? 0 : (*this)[2][2] * v[2]);

			matrix[3][0] = (*this)[3][0];
			matrix[3][1] = (*this)[3][1];
			matrix[3][2] = (*this)[3][2];

			matrix[3] = (*this)[3];
			return matrix;
		}

		constexpr _matrix4x4 rotate(f32_t angle, const fan::vec3& v) const {
			const f32_t a = angle;
			const f32_t c = cos(a);
			const f32_t s = sin(a);
			fan::vec3 axis(fan_3d::math::normalize(v));
			fan::vec3 temp(axis * (1.0f - c));

			_matrix4x4 rotation{};
			rotation[0][0] = c + temp[0] * axis[0];
			rotation[0][1] = temp[0] * axis[1] + s * axis[2];
			rotation[0][2] = temp[0] * axis[2] - s * axis[1];

			rotation[1][0] = temp[1] * axis[0] - s * axis[2];
			rotation[1][1] = c + temp[1] * axis[1];
			rotation[1][2] = temp[1] * axis[2] + s * axis[0];

			rotation[2][0] = temp[2] * axis[0] + s * axis[1];
			rotation[2][1] = temp[2] * axis[1] - s * axis[0];
			rotation[2][2] = c + temp[2] * axis[2];

			_matrix4x4 matrix{};
			matrix[0][0] = ((*this)[0][0] * rotation[0][0]) + ((*this)[1][0] * rotation[0][1]) + ((*this)[2][0] * rotation[0][2]);
			matrix[1][0] = ((*this)[0][1] * rotation[0][0]) + ((*this)[1][1] * rotation[0][1]) + ((*this)[2][1] * rotation[0][2]);
			matrix[2][0] = ((*this)[0][2] * rotation[0][0]) + ((*this)[1][2] * rotation[0][1]) + ((*this)[2][2] * rotation[0][2]);

			matrix[0][1] = ((*this)[0][0] * rotation[1][0]) + ((*this)[1][0] * rotation[1][1]) + ((*this)[2][0] * rotation[1][2]);
			matrix[1][1] = ((*this)[0][1] * rotation[1][0]) + ((*this)[1][1] * rotation[1][1]) + ((*this)[2][1] * rotation[1][2]);
			matrix[2][1] = ((*this)[0][2] * rotation[1][0]) + ((*this)[1][2] * rotation[1][1]) + ((*this)[2][2] * rotation[1][2]);

			matrix[0][2] = ((*this)[0][0] * rotation[2][0]) + ((*this)[1][0] * rotation[2][1]) + ((*this)[2][0] * rotation[2][2]);
			matrix[1][2] = ((*this)[0][1] * rotation[2][0]) + ((*this)[1][1] * rotation[2][1]) + ((*this)[2][1] * rotation[2][2]);
			matrix[2][2] = ((*this)[0][2] * rotation[2][0]) + ((*this)[1][2] * rotation[2][1]) + ((*this)[2][2] * rotation[2][2]);

			matrix[3] = (*this)[3];

			return matrix;
		}

	private:


	};

	using matrix4x4 = _matrix4x4<cf_t>;
	using matrix4x4ui = _matrix4x4<uintptr_t>;
	using mat4x4 = matrix4x4;
	using mat4x4ui = matrix4x4ui;
	using mat4 = mat4x4;
	using mat2x2 = _matrix2x2<cf_t>;
	using mat2x2ui = _matrix2x2<uintptr_t>;
	using mat2 = mat2x2;
	using mat2ui = mat2x2ui;

}