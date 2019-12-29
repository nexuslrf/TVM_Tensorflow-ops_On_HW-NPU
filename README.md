# To-Do

Roadmap:

- [ ] Build Computation Graph (By Dec 30th)
- [ ] Write Test Case (By Jan 1st)
- [ ] Scheduling Optimization (By Jan 3rd)
- [ ] Test on NPU? 

-------

Current State: Build Computation Graph

* [ ] CTCBeamSearchDecoder: tensorflow/core/kernels/ctc_decoder_ops.cc
* [ ] CTCGreedyDecoder: tensorflow/core/kernels/ctc_decoder_ops.cc
* [ ] CTCLoss: tensorflow/core/kernels/ctc_loss_ops.cc
* [ ] Add
* [ ] Atan
* [ ] Exp
* [ ] SegmentMean
* [ ] SegmentMin
* [ ] SegmentProd
* [ ] SegmentSum

-----

Useful Notes: [TVM-Learning](TVM-Learning.md)

頑張って！

---------

Problems:

1. 我们只需要用TVM(TBE 原语)进行开发？
2. 可否提供一个TVM开发的完整项目模版，样例项目中只有DSL和TIK。文档中的提到的TVM开发样例也只是代码片段不够完整。
3. NPU上，数据必须手动cache read/write到UB吗，指令必须用emit_insn手动映射吗？如果不这么做会怎么样？