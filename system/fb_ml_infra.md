1. Training NN
   1. SGD
   1. Batch can use data ||
   1. Requires sync
   1. Do you need specialized interconnects?
      1. Ethernet based networking to be sufficient
         1. near-linear scaling
      1. if network BW is too low s.t. parameter sync takes more time than gradient computation, data || benefits diminish
   1. SYNCHRONIZATION
      1. replicas see same state
   1. CONSISTENCY
      1. generate correct updates
   1. PERFORMANCE
      1. scales sub-linearly
1. MODEL ||
   1. break up into layers
      1. increase in end-to-end latency
    