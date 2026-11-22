// First frontend target: whole-array computation, before GPU thread mapping.
module {
  func.func @compute(%v0: tensor<128xf32>, %v1: tensor<128xf32>) -> tensor<128xf32> {
    %v2 = arith.addf %v0, %v1 : tensor<128xf32>
    %v3 = arith.mulf %v2, %v2 : tensor<128xf32>
    func.return %v3 : tensor<128xf32>
  }
}
