# pyright: reportInvalidTypeForm=false
import math
from collections.abc import Iterable, Sequence
from typing import List, Optional

try:
    import cython
except ImportError:

    class _CythonStub:
        int = int
        double = float
        bint = bool
        Py_ssize_t = int

        @staticmethod
        def cfunc(func):
            return func

        inline = cfunc

        @staticmethod
        def boundscheck(_):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def wraparound(_):
            def decorator(func):
                return func

            return decorator

    cython = _CythonStub()

BitList = List[int]
Trellis = List[List[int]]


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def hamming_distance(l1: BitList, l2: BitList) -> cython.int:
    limit: cython.Py_ssize_t = min(len(l1), len(l2))
    dist: cython.int = 0
    idx: cython.Py_ssize_t

    for idx in range(limit):
        a: cython.int = l1[idx]
        b: cython.int = l2[idx]
        if a != b and a != -1 and b != -1:
            dist += 1

    return dist


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def soft_distance(code_bits: BitList, llrs: List[float]) -> cython.double:
    limit: cython.Py_ssize_t = min(len(code_bits), len(llrs))
    dist: cython.double = 0.0
    idx: cython.Py_ssize_t

    for idx in range(limit):
        llr: cython.double = llrs[idx]
        contrib: cython.double = llr if code_bits[idx] else -llr
        dist += contrib

    return dist


class Viterbi:
    _constraint: cython.int
    _polynomials: BitList
    _puncpat: Optional[BitList]
    _outputs: List[BitList]
    _n_parity_bits: cython.int

    def __init__(
        self,
        constraint: cython.int,
        polynomials: Sequence[int],
        puncpat: Optional[Sequence[int]] = None,
    ) -> None:
        self._constraint = constraint
        self._polynomials = list(polynomials)
        self._puncpat = list(puncpat) if puncpat is not None else None
        self._outputs = [[] for _ in range(1 << self._constraint)]
        self._n_parity_bits = len(self._polynomials)

        for i in range(1 << self._constraint):
            for p in self._polynomials:
                self._outputs[i].append(int.bit_count(i & p) % 2)

    def _puncture(self, bits: BitList) -> BitList:
        if self._puncpat is None:
            return bits

        puncpat_len: cython.int = len(self._puncpat)
        return [bit for i, bit in enumerate(bits) if self._puncpat[i % puncpat_len] == 1]

    def _depuncture(self, bits: BitList, b_erase: cython.double = -1.0) -> list:
        if self._puncpat is None:
            return bits

        depunctured: list = []
        it = iter(bits)
        while True:
            for flag in self._puncpat:
                if flag == 1:
                    try:
                        depunctured.append(next(it))
                    except StopIteration:
                        return depunctured
                else:
                    depunctured.append(b_erase)

    def encode(self, bits: Iterable[int]) -> BitList:
        output: BitList = []
        state: cython.int = 0
        b: cython.int

        for b in bits:
            state = (state >> 1) | b << (self._constraint - 1)
            output.extend(self._outputs[state])

        if self._puncpat is not None:
            return self._puncture(output)
        else:
            return output

    def decode(self, bits: Sequence, soft: bool = False) -> BitList:
        b_erase: cython.double = 0.0 if soft else -1.0
        is_soft: cython.bint = soft

        if self._puncpat is not None:
            bits = self._depuncture(list(bits), b_erase=b_erase)
        else:
            bits = list(bits)

        trellis: Trellis = []
        max_state: cython.int = 1 << (self._constraint - 1)
        path_metrics: List[float] = [0.0 if i == 0 else math.inf for i in range(max_state)]

        n_steps: cython.int = math.ceil(len(bits) / self._n_parity_bits)
        step: cython.int
        for step in range(n_steps):
            trellis.append([])
            cur_path_metrics: List[float] = []

            start: cython.int = step * self._n_parity_bits
            stop: cython.int = (step + 1) * self._n_parity_bits
            cur_bits = bits[start:stop]

            if len(cur_bits) < self._n_parity_bits:
                # pad erase bits
                cur_bits += [b_erase] * (self._n_parity_bits - len(cur_bits))

            cur: cython.int
            mask: cython.int = (1 << (self._constraint - 1)) - 1

            for cur in range(max_state):
                prev1: cython.int = (cur << 1) | 0
                prev2: cython.int = (cur << 1) | 1

                if is_soft:
                    pm1 = soft_distance(self._outputs[prev1], cur_bits) + path_metrics[prev1 & mask]
                    pm2 = soft_distance(self._outputs[prev2], cur_bits) + path_metrics[prev2 & mask]
                else:
                    pm1 = hamming_distance(self._outputs[prev1], cur_bits) + path_metrics[prev1 & mask]
                    pm2 = hamming_distance(self._outputs[prev2], cur_bits) + path_metrics[prev2 & mask]

                if pm1 < pm2:
                    trellis[step].append(prev1 & mask)
                    cur_path_metrics.append(pm1)
                else:
                    trellis[step].append(prev2 & mask)
                    cur_path_metrics.append(pm2)

            path_metrics = cur_path_metrics

        # traceback
        out: BitList = []
        state: cython.int = path_metrics.index(min(path_metrics))

        i: cython.int
        for i in reversed(range(len(trellis))):
            out.append(state >> (self._constraint - 2))
            state = trellis[i][state]

        return out[::-1]


__all__ = ["Viterbi"]
