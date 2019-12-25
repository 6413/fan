#pragma once
#include <iostream>

#define ALLOC_BUFFER 0xff

template <typename _Type>
class Alloc {
public:
	Alloc() : _Last(0), _Size(0), _Data(NULL), _Current(0) {}

	Alloc(size_t _Reserve) : _Last(_Reserve - 1), _Current(0) {
		resize(_Reserve);
	}
	~Alloc() {}
	constexpr _Type& operator[](size_t _Where) {
		return _Data[_Where];
	}
	constexpr _Type operator[](size_t _Where) const {
		return _Data[_Where];
	}
	void push_back(_Type _Value) {
		if (_Size > _Current) {
			_Data[_Current] = _Value;
		}
		if (this->empty()) {
			_Data = new _Type[++_Size];
			_Data[_Last] = _Value;
			_Last = _Size;
			_Current++;
			return;
		}
		if (_Size <= _Current) {
			_Type* _Temp = new _Type[_Size];
			this->copy(_Data, _Temp, _Last);
			_Size += ALLOC_BUFFER;
			_Data = new _Type[_Size];
			this->copy(_Temp, _Data, _Last);
			delete[] _Temp;
		}
		_Data[_Current] = _Value;
		_Last = _Size;
		_Current++;
	}
	constexpr void copy(const _Type* _Src, _Type* _Dest, size_t _Buffer) {
		int _Index = 0;
		for (; _Index != _Buffer; ++_Index) {
			_Dest[_Index] = _Src[_Index];
		}
	}
	constexpr size_t size() const {
		return this->_Current;
	}
	constexpr bool empty() const {
		return !this->_Size;
	}
	constexpr size_t begin() const {
		return 0;
	}
	constexpr size_t end() const {
		return this->_Size;
	}
	constexpr _Type* data() const {
		return _Data;
	}
	constexpr void resize(size_t _Reserve) {
		_Data = new _Type[_Reserve];
		_Size = _Reserve;
	}
	constexpr void free() const {
		delete[] _Data;
	}
private:
	size_t _Last;
	size_t _Size;
	size_t _Current;
	_Type* _Data;
};