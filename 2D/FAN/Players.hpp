#pragma once
#include "Texture.hpp"

constexpr GroupId GetEnemy() {
	return GroupId::Enemy;
}

template <typename T>
constexpr arrayIndex GetLocalPlayer(std::vector<T>& entity) {
	for (int i = 0; i < entity.size(); i++) {
		if (entity[i].groupId == GroupId::LocalPlayer) {
			return i;
		}
		if (entity[i].groupId == GroupId::NotAssigned) {
			return i;
		}
	}
	return NULL;
}

template <typename T>
arrayIndex GetEnemy(std::vector<T>& entity) {
	for (int i = 0; i < entity.size(); i++) {

		if (entity[i].groupId == GroupId::Enemy) {
			return i;
		}
	}
	return NULL;
}