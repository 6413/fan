#pragma once
#include <iostream>

template <typename _Type>
class Alloc;

template <typename _Type>
class iterator {
public:
	iterator() : _Ptr(0) {}
	iterator(_Type* _Temp) : _Ptr(_Temp) {}
	constexpr bool operator!=(const iterator& _It) {
		return _It._Ptr != _Ptr;
	}
	constexpr iterator operator++() {
		++_Ptr;
		return *this;
	}
	constexpr iterator operator--() {
		--_Ptr;
		return *this;
	}
	constexpr iterator operator-(iterator<_Type> _Index) {
		_Ptr = (_Type*)(_Ptr - _Index._Ptr);
		return *this;
	}
	constexpr iterator operator-(size_t _Index) {
		_Ptr = (_Type*)(_Ptr - _Index);
		return *this;
	}
	constexpr _Type& operator*() const {
		return *_Ptr;
	}
	constexpr iterator operator+(_Type _Value) const {
		return _Ptr + _Value;
	}
protected:
	_Type* _Ptr;
};

template <typename _Ty, typename _Ty2>
_Ty* operator*(_Ty* _Arr, _Ty2 _Val) {
	_Ty* Arr[ArrLen(_Arr)];
	for (int i = 0; i < ArrLen(_Arr); i++) {
		Arr[i] = _Arr[i] * _Val;
	}
	return Arr;
}

template <typename _Ty>
class Alloc : public iterator<_Ty> {
public:
	Alloc() : _Cur(0), _Allocated(0) {
#ifdef _DBT
		_Cur = 1;
#endif
	}

	Alloc(std::size_t _Reserve) : _Cur(0) {
		resize(_Reserve);
	}

	Alloc(std::size_t _Reserve, _Ty _InIt) : _Cur(0) {
		resize(_Reserve);
		for (std::size_t _I = 0; _I < _Reserve; _I++) {
			push_back(_InIt);
		}
	}

	~Alloc() {
		if (_Data) {
			//_Data = 0;
//			delete[] _Data;
		}
	}

	_Ty operator[](std::size_t _Idx) const {
		return _Data[_Idx];
	}

	_Ty& operator[](std::size_t _Idx) {
		return _Data[_Idx];
	}

	void free_to_max() {
		if (_Allocated > _Cur) {
			_Ty* _Temp = new _Ty[_Cur];
			std::copy(_Data, _Data + _Cur, _Temp);
			delete[] _Data;
			_Allocated = _Cur;
			_Data = new _Ty[_Cur];
			std::copy(_Temp, _Temp + _Cur, _Data);
			delete[] _Temp;
		}
	}

	template <class... _Valty>
	decltype(auto) emplace_back(_Valty&&... _Val) {
		if (_Cur >= _Allocated) {
			resize(_Cur + 1);
		}
		*end() = _Ty(std::forward<_Valty>(_Val)...);
		_Cur++;
		return _Data;
	}

	void push_back(const _Ty& _Val) {
		emplace_back(_Val);
	}

	void push_back(_Ty&& _Val) {
		emplace_back(std::move(_Val));
	}

	void resize(std::size_t _Reserve) {
		if (!_Cur) {
			_Data = new _Ty[_Reserve];
			_Allocated = _Reserve;
		}
		else if (_Reserve > _Allocated) {
			_Ty* _Temp = new _Ty[_Allocated];
			std::copy(_Data, _Data + _Allocated, _Temp);
			delete[] _Data;
			_Data = new _Ty[_Reserve + _Buffer];
			std::copy(_Temp, _Temp + _Allocated, _Data);
			_Allocated = _Reserve + _Buffer;
			delete[] _Temp;
		}
	}

	iterator<_Ty> begin() const {
		return _Data;
	}

	iterator<_Ty> end() const {
		return _Data + _Cur;
	}

	std::size_t current() const {
		return _Cur;
	}

	bool empty() const {
		return !_Allocated;
	}

	constexpr void erase(size_t _Idx) {
		for (size_t _I = _Idx; _I < _Cur - 1; _I++) {
			_Data[_I] = _Data[_I + 1];
		}
		resize(--_Cur);
	}

	_Ty* data() const {
		return _Data;
	}

	std::size_t size() const {
		return _Allocated;
	}

	std::size_t _Buffer = 0x1;
private:
	_Ty* _Data;
	std::size_t _Cur;
	std::size_t _Allocated;
};