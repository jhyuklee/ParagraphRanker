import numpy as np


def write_vint(i):
    var_int_bytes = bytes()
    while (i & ~0x7F) != 0:
        var_int_bytes += bytes([((i & 0x7f) | 0x80)])
        i = np.right_shift(i, 7)
    var_int_bytes += bytes([i])
    return var_int_bytes


class BinaryReader(object):
    def __init__(self, binary):
        self.binary = binary
        self.pos = 0

    def read_vint(self):
        b = self.binary[self.pos]
        self.pos += 1
        if 128 > b >= 0:  # unsigned
            return b
        i = b & 0x7F
        b = self.binary[self.pos]
        self.pos += 1
        i |= (b & 0x7F) << 7
        if 128 > b >= 0:  # unsigned
            return i
        b = self.binary[self.pos]
        self.pos += 1
        i |= (b & 0x7F) << 14
        if 128 > b >= 0:  # unsigned
            return i
        b = self.binary[self.pos]
        self.pos += 1
        i |= (b & 0x7F) << 21
        if 128 > b >= 0:  # unsigned
            return i
        b = self.binary[self.pos]
        self.pos += 1
        # Warning: the next ands use 0x0F / 0xF0 - beware copy/paste errors:
        i |= (b & 0x0F) << 28
        if (b & 0xF0) == 0:
            return i
        raise RuntimeError("Invalid vInt detected (too many bits)")


def test():
    vint_ex1 = write_vint(1)
    print(list(vint_ex1))
    br = BinaryReader(vint_ex1)
    print(br.read_vint())

    vint_ex2 = write_vint(128)
    print(list(vint_ex2))
    br = BinaryReader(vint_ex2)
    print(br.read_vint())

    vint_ex3 = write_vint(16384)
    print(list(vint_ex3))
    br = BinaryReader(vint_ex3)
    print(br.read_vint())


if __name__ == '__main__':
    test()
