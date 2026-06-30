/**
 * @file
 */

#pragma once

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

namespace core {

/**
 * @brief Allows to store boolean values in a compact bit buffer with runtime-determined size
 * @ingroup Collections
 */
class DynamicBitSet {
public:
	using Type = uint64_t;

private:
	static constexpr size_t bitsPerValue = sizeof(Type) * CHAR_BIT;

	static size_t requiredElements(size_t bits) {
		return bits == 0 ? 0 : (bits + bitsPerValue - 1) / bitsPerValue;
	}

	Type *_buffer = nullptr;
	size_t _size = 0; // number of bits

public:
	DynamicBitSet() = default;

	explicit DynamicBitSet(size_t size);
	~DynamicBitSet();
	DynamicBitSet(const DynamicBitSet &other);
	DynamicBitSet &operator=(const DynamicBitSet &other);
	DynamicBitSet(DynamicBitSet &&other) noexcept;
	DynamicBitSet &operator=(DynamicBitSet &&other) noexcept;

	inline size_t bits() const {
		return _size;
	}

	void resize(size_t newSize);

	void fill();
	void invert();
	bool hasBitsSet() const;

	void clear();

	size_t bytes() const;

	inline void set(size_t idx, bool value) {
		if (idx >= _size) {
			return;
		}
		const size_t arrayIdx = idx / bitsPerValue;
		const size_t elementIdx = idx % bitsPerValue;
		Type &ref = _buffer[arrayIdx];
		if (value) {
			ref |= (Type(1) << elementIdx);
		} else {
			ref &= ~(Type(1) << elementIdx);
		}
	}

	inline bool operator[](size_t idx) const {
		if (idx >= _size) {
			return false;
		}
		const size_t arrayIdx = idx / bitsPerValue;
		const size_t elementIdx = idx % bitsPerValue;
		const Type &ref = _buffer[arrayIdx];
		const Type mask = (Type(1) << elementIdx);
		return (ref & mask) != 0;
	}
	bool operator==(const DynamicBitSet &other) const;
	bool operator!=(const DynamicBitSet &other) const;

	inline const Type *buffer() const {
		return _buffer;
	}

	inline size_t words() const {
		return requiredElements(_size);
	}
};

} // namespace core
