"""Manifest and argument checks that do not require a GPU."""

import ctypes as ct
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tensor_runtime import TensorProgram, fingerprint, read_manifest


class TensorRuntimeTests(unittest.TestCase):
    def test_manifest_requires_matching_ptx_and_valid_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            ptx = Path(directory) / "test.ptx"
            image = b"test PTX bytes"
            ptx.write_bytes(image)
            manifest = {
                "version": 1, "dtype": "float32", "ptx_fnv1a64": fingerprint(image),
                "inputs": [0], "buffers": [[3, 2], [3, 2]], "output": 1,
                "kernels": [{"name": "test", "buffers": [0, 1],
                             "grid": [1, 1, 1], "block": [128, 1, 1]}],
            }
            metadata = ptx.with_suffix(".json")
            metadata.write_text(json.dumps(manifest))
            self.assertEqual(read_manifest(ptx)[1]["buffers"], [[3, 2], [3, 2]])
            ptx.write_bytes(image + b" changed")
            with self.assertRaisesRegex(ValueError, "do not match"):
                read_manifest(ptx)
            ptx.write_bytes(image)
            manifest["kernels"][0]["buffers"] = [99]
            metadata.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "buffer IDs"):
                read_manifest(ptx)

    def test_wrong_arguments_fail_before_any_cuda_access(self):
        program = object.__new__(TensorProgram)
        program.context = ct.c_void_p(1)
        program.input_shapes = [(3, 2)]
        for arguments in [(), (np.zeros((3, 2), dtype=np.float64),),
                          (np.zeros((2, 3), dtype=np.float32),)]:
            with self.assertRaises(ValueError):
                program(*arguments)
        program.context = ct.c_void_p()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            program(np.zeros((3, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
