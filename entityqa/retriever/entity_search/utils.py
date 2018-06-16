import numpy as np


def get_vint_bytes(i):
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
    vint_ex1 = get_vint_bytes(1 << 0)
    print(list(vint_ex1), vint_ex1.hex())
    br = BinaryReader(vint_ex1)
    print(br.read_vint())

    vint_ex2 = get_vint_bytes(1 << 7)
    print(list(vint_ex2), vint_ex2.hex())
    br = BinaryReader(vint_ex2)
    print(br.read_vint())

    vint_ex3 = get_vint_bytes(1 << 14)
    print(list(vint_ex3), vint_ex3.hex())
    br = BinaryReader(vint_ex3)
    print(br.read_vint())

    vint_ex4 = get_vint_bytes(1 << 21)
    print(list(vint_ex4), vint_ex4.hex())
    br = BinaryReader(vint_ex4)
    print(br.read_vint())

    vint_ex5 = get_vint_bytes((1 << 28) - 1)  # 4-bytes max
    print(list(vint_ex5), vint_ex5.hex())
    br = BinaryReader(vint_ex5)
    print(br.read_vint())

    # 11614	2182	88136	1385	21282267	178043	21282268	6078
    print(get_vint_bytes(11614).hex(), get_vint_bytes(2182).hex(),
          get_vint_bytes(88136).hex(), get_vint_bytes(1385).hex(),
          get_vint_bytes(21282267).hex(),
          get_vint_bytes(178043).hex(), get_vint_bytes(21282268).hex(),
          get_vint_bytes(6078).hex())

    # join
    eids = [70845,	516396,	501989,	516407]
    bslist = []
    bts = bytes()
    sz = get_vint_bytes((len(eids) << 1) + 1)
    bts += sz
    bslist.append(sz)
    for eid in eids:
        bs = get_vint_bytes(eid)
        print(bs)
        bslist.append(bs)
        bts += bs
    print(bslist)
    print(bts, len(bts))
    print(bytearray(bts), bts.decode('utf-8', 'backslashreplace'))
    print(''.join(str(i) for i in list(bts)))
    print(bts.hex(), len(bts))
    print(b''.join(bslist))

    print(BinaryReader(b'\xd8\x95\x02').read_vint())
    print(BinaryReader(b'\xab\xa1\x10').read_vint())


if __name__ == '__main__':
    test()
