#pragma once
#include <array>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <numeric>

template <typename T>
constexpr f_t MinFloat_fr(const T a) {
	f_t Min = FLT_MAX;
	ui_t i = 0;
	for (; i < a.size(); i++) {
		if (a[i] < Min)
			Min = a[i];
	}
	return Min;
}

template <typename T>
constexpr f_t MaxFloat_fr(const T a) {
	f_t Max = -FLT_MAX;
	ui_t i = 0;
	for (; i < a.size(); i++) {
		if (a[i] > Max)
			Max = a[i];
	}
	return Max;
}

template <typename type, std::size_t Rows, std::size_t Cols>
struct matrix;

template <typename type, std::size_t Rows>
struct list;

template <typename type, std::size_t Rows, std::size_t Cols = 1>
using da_t = std::conditional_t<Cols == 1, list<type, Rows>, matrix<type, Rows, Cols>>;

template <typename T, typename type, std::size_t rows>
concept is_da_t = std::is_same_v<T, da_t<type, rows>>;

template <typename T, typename type>
concept is_vector_t = std::is_same_v<T, _vec2<type>> || std::is_same_v<T, _vec3<type>> || std::is_same_v<T, _vec4<type>>;

template <typename T, typename type, std::size_t rows>
concept is_not_da_t = !is_da_t<T, type, rows>;

template <typename T, typename type>
concept is_not_vector_t = !is_vector_t<T, type>;

template <typename T, typename T2, typename _Func>
constexpr void foreach(T src_begin, T src_end, T2 dst_begin, _Func function) {
	auto dst = dst_begin;
	for (auto it = src_begin; it != src_end; ++it, ++dst) {
		*dst = function(*it);
	}
}

template <typename type, std::size_t rows>
struct list : public std::array<type, rows> {

	using array_type = std::array<type, rows>;

	template <typename T>
	constexpr list(const _vec2<T>& vector) {
		for (int i = 0; i < std::min(rows, vector.size()); i++) {
			this->operator[](i) = vector[i];
		}
	}

	template <typename T>
	constexpr list(const _vec3<T>& vector) {
		for (int i = 0; i < std::min(rows, vector.size()); i++) {
			this->operator[](i) = vector[i];
		}
	}

	template <typename T>
	constexpr list(const _vec4<T>& vector) {
		for (int i = 0; i < std::min(rows, vector.size()); i++) {
			this->operator[](i) = vector[i];
		}
	}

	template <typename ...T>
	constexpr list(T... x) : std::array<type, rows>{ (type)x... } {}

	template <typename T>
	constexpr list(T value) : std::array<type, rows>{0} {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) = value;
		}
	}

	template <typename T, std::size_t array_n>
	constexpr list(const list<T, array_n>& list) {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) = list[i];
		}
	}

	constexpr auto operator++() noexcept {
		return &this->_Elems + 1;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator+(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) + _list[i];
		}
		return calculation_list;
	}

	template <is_not_da_t<type, rows> T>
	constexpr list operator+(T value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) + value;
		}
		return list;
	}

	template <typename T, std::size_t rows_>
	constexpr list operator+=(const list<T, rows_>& value) noexcept {
		//static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) += value[i];
		}
		return *this;
	}


	template <is_not_da_t<type, rows> T>
	constexpr list operator+=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) += value;
		}
		return *this;
	}

	constexpr list operator-() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = -this->operator[](i);
		}
		return l;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator-(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) - _list[i];
		}
		return calculation_list;
	}

	template <is_not_da_t<type, rows> T>
	constexpr list operator-(T value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) - value;
		}
		return list;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator-=(const list<T, list_n>& value) noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) -= value[i];
		}
		return *this;
	}

	template <is_not_da_t<type, rows> T>
	constexpr list operator-=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) -= value;
		}
		return *this;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator*(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) * _list[i];
		}
		return calculation_list;
	}

	constexpr list operator*(type value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) * value;
		}
		return list;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator*=(const list<T, list_n>& value) noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) *= value[i];
		}
		return *this;
	}

	template <is_not_da_t<type, rows> T>
	constexpr list operator*=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) *= value;
		}
		return *this;
	}


	template <typename T, std::size_t list_n>
	constexpr list operator/(const list<T, list_n>& _list) const noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		list calculation_list;
		for (int i = 0; i < rows; i++) {
			calculation_list[i] = this->operator[](i) / _list[i];
		}
		return calculation_list;
	}

	template <is_not_da_t<type, rows> T>
	constexpr list operator/(T value) const noexcept {
		list list;
		for (int i = 0; i < rows; i++) {
			list[i] = this->operator[](i) / value;
		}
		return list;
	}

	template <typename T, std::size_t list_n>
	constexpr list operator/=(const list<T, list_n>& value) noexcept {
		static_assert(rows >= list_n, "second list is bigger than first");
		for (int i = 0; i < rows; i++) {
			this->operator[](i) /= value[i];
		}
		return *this;
	}

	template <is_not_da_t<type, rows> T>
	constexpr list operator/=(T value) noexcept {
		for (int i = 0; i < rows; i++) {
			this->operator[](i) /= value;
		}
		return *this;
	}

	template <is_not_da_t<type, rows> T>
	constexpr auto operator%(T value) {
		list l;
		for (int i = 0; i < rows; i++) {
			l = fmodf(this->operator[](i), value);
		}
		return l;
	}

	constexpr bool operator<(const list<type, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) < list_[i]) {
				return true;
			}
		}
		return false;
	}

	constexpr bool operator<=(const list<type, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) <= list_[i]) {
				return true;
			}
		}
		return false;
	}

	template <is_not_da_t<type, rows> T>
	constexpr bool operator==(T value) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) != value) {
				return false;
			}
		}
		return true;
	}

	constexpr bool operator==(const list<type, rows>& list_) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) != list_[i]) {
				return false;
			}
		}
		return true;
	}

	template <is_not_da_t<type, rows> T>
	constexpr bool operator!=(T value) {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i) == value) {
				return false;
			}
		}
		return true;
	}

	template <typename T>
	constexpr bool operator!=(const list<T, rows>& list_) {
		for (int i = 0; i < rows; i++) {
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
		for (int i = 0; i < rows; i++) {
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
		for (int i = 0; i < rows; i++) {
			l[i] = std::abs(this->operator[](i));
		}
		return l;
	}

	constexpr list<type, 2> floor() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = std::floor(this->operator[](i));
		}
		return l;
	}

	constexpr list<type, 2> ceil() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = (this->operator[](i) < 0 ? -std::ceil(-this->operator[](i)) : std::ceil(this->operator[](i)));
		}
		return l;
	}

	constexpr list<type, 2> round() const noexcept {
		list l;
		for (int i = 0; i < rows; i++) {
			l[i] = std::round(-this->operator[](i));
		}
		return l;
	}

	constexpr type pmax() const noexcept {
		decltype(*this) list = this->abs();
		auto biggest = std::max_element(list.begin(), list.end());
		if (this->operator[](biggest - list.begin()) < 0) {
			return -*biggest;
		}
		return *biggest;
	}

	constexpr auto gfne() const noexcept {
		for (int i = 0; i < rows; i++) {
			if (this->operator[](i)) {
				return this->operator[](i);
			}
		}
		return type();
	}
};

template <typename T>
constexpr bool dcom_fr(uint_t n, T x, T y) noexcept {
	switch (n) {
	case 0: {
		return x < y;
	}
	case 1: {
		return x > y;
	}
	}
	return false;
}

template <typename type, std::size_t Rows, std::size_t Cols>
struct matrix {

	list<type, Cols> m[Rows];

	using matrix_type = matrix<type, Rows, Cols>;
	using value_type = list<type, Cols>;

	static constexpr std::size_t rows = Rows;
	static constexpr std::size_t cols = Cols;

	constexpr matrix() : m{ 0 } { }

	template <typename _Type, std::size_t cols, template <typename, std::size_t> typename... _List>
	constexpr matrix(const _List<_Type, cols>&... list_) : m{ 0 } {
		//static_assert(sizeof...(list_) >= cols, "too many initializers");
		int i = 0;
		auto auto_type = std::get<0>(std::forward_as_tuple(list_...));
		((((decltype(auto_type)*)m)[i++] = list_), ...);
	}

	template <typename T>
	constexpr matrix(const matrix<T, rows, cols>& matrix_) : m{ 0 } {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				m[i][j] = matrix_[i][j];
			}
		}
	}

	template <typename..._Type>
	constexpr matrix(_Type... value) {
		if constexpr (sizeof...(value) == cols * rows) {
			int init = 0;
			((((type*)m)[init++] = value), ...);
		}
	}

	template <typename T>
	constexpr matrix(T value) : m{ 0 } {
		for (int i = 0; i < rows && i < cols; i++) {
			m[i][i] = value;
		}
	}

	template <template <typename> typename... _Vec_t, typename _Type>
	matrix(const _Vec_t<_Type>&... vector) : m{ 0 } {
		int i = 0;
		auto auto_type = std::get<0>(std::forward_as_tuple(vector...));
		((((decltype(auto_type)*)m)[i++] = vector), ...);
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr matrix_type operator+(const matrix<T, Rows2, Cols2>& matrix) const noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] + matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator+(const list<T, da_t_n>& da_t) const noexcept { // list
		matrix_type _matrix;
		static_assert(cols >= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] + da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator+(T value) const noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] + value;
		}
		return _matrix;
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr auto operator+=(const matrix<T, Rows2, Cols2>& matrix) noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] += matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator+=(const list<T, da_t_n>& da_t) noexcept { // list
		matrix_type _matrix;
		static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] += da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator+=(T value) noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] += value;
		}
		return _matrix;
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr matrix_type operator-(const matrix<T, Rows2, Cols2>& matrix) const noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] - matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator-(const list<T, da_t_n>& da_t) const noexcept { // list
		matrix_type _matrix;
		static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] - da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator-(T value) const noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] - value;
		}
		return _matrix;
	}

	template <typename T, std::size_t Rows2, std::size_t Cols2>
	constexpr matrix_type operator-=(const matrix<T, Rows2, Cols2>& matrix) noexcept { // matrix
		matrix_type _matrix;
		static_assert(Cols2 <= Cols, "Colums of the second matrix is bigger than first");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] -= matrix[i];
		}
		return _matrix;
	}

	template <typename T, std::size_t da_t_n>
	constexpr matrix_type operator-=(const list<T, da_t_n>& da_t) noexcept { // list
		matrix_type _matrix;
		static_assert(cols <= da_t_n, "list is bigger than the matrice's Rows");
		for (int i = 0; i < rows; i++) {
			_matrix[i] = m[i] -= da_t;
		}
		return _matrix;
	}

	template <typename T>
	constexpr matrix_type operator-=(T value) noexcept { // basic value
		matrix_type _matrix;
		for (int i = 0; i < rows; i++) {
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

	constexpr matrix_type operator*(const matrix<type, Rows, Cols>& _Lhs) noexcept(false) {
		if (Rows != Cols) {
			throw("first matrix rows must be same as second's colums");
		}
		for (int _I = 0; _I < Rows; _I++) {
			for (int _J = 0; _J < Cols; _J++) {
				type _Value = 0;
				for (int _K = 0; _K < Cols; _K++) {
					_Value += m[_I][_K] * _Lhs[_K][_J];
				}
				m[_I][_J] = _Value;
			}
		}
		return *this;
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
		for (int i = 0; i < Cols; i++) {
			for (int j = 0; j < Rows; j++) {
				m[i][j] = fmodf(m[i][j], value);
			}
		}
		return *this;
	}

	constexpr bool operator==(const matrix<type, rows, cols>& list_) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
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
		for (int i = 0; i < rows; i++) {
			m[i].print();
		}
	}

	list<type, Cols>* begin() noexcept {
		return &m[0];
	}

	list<type, Cols>* end() noexcept {
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
		list<type, rows> averages;
		for (int i = 0; i < rows; i++) {
			averages += m[i];
		}
		return averages / rows;
	}


	constexpr auto vector() noexcept {
		return std::vector<da_t<type, cols>>(this->begin(), this->end());
	}

	constexpr auto size() const noexcept {
		return rows;
	}
};

template <typename T>
using _mat2x2 = matrix<T, 2, 2>;
template <typename T>
using _mat2x3 = matrix<T, 2, 3>;
template <typename T>
using _mat3x2 = matrix<T, 3, 2>;
template <typename T>
using _mat4x2 = matrix<T, 4, 2>;
template <typename T>
using _mat3x3 = matrix<T, 3, 3>;
template <typename T>
using _mat4x4 = matrix<T, 4, 4>;

using mat2x2 = _mat2x2<f32_t>;
using mat2x3 = _mat2x3<f32_t>;
using mat3x2 = _mat3x2<f32_t>;
using mat4x2 = _mat4x2<f32_t>;
using mat3x3 = _mat3x3<f32_t>;
using mat4x4 = _mat4x4<f32_t>;

using mat2 = mat2x2;
using mat3 = mat3x3;
using mat4 = mat4x4;


template <typename T, std::size_t rows>
std::ostream& operator<<(std::ostream& os, const list<T, rows> list_) noexcept
{
	for (int i = 0; i < rows; i++) {
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
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			os << da_t_[i][j] << ' ';
		}
	}
	return os;
}

template <typename ...Args>
constexpr void LOG(const Args&... args) {
	((std::cout << args << " "), ...) << '\n';
}