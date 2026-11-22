// Handwritten backend reference: 128 floats, one block of 128 threads.
// Kernel ABI after lowering: three device pointers (x, y, output).
gpu.module @kernels {
  gpu.func @compute(%x: memref<128xf32>, %y: memref<128xf32>,
                    %out: memref<128xf32>) kernel {
    %i = gpu.thread_id x
    %a = memref.load %x[%i] : memref<128xf32>
    %b = memref.load %y[%i] : memref<128xf32>
    %sum = arith.addf %a, %b : f32
    %square = arith.mulf %sum, %sum : f32
    memref.store %square, %out[%i] : memref<128xf32>
    gpu.return
  }
}
