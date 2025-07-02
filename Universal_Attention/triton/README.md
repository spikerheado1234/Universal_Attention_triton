# Triton implementations for unit testing
Cumulative sum, broadcasted matrix multiplication, softmax, ...

Lessons learned:
* Triton is really bad at inter-block synchronization. Avoid doing so.
* My original plan for implementing the kernel was through chunking and distributing the computation. Clearly this plan does not levitate the memory bottleneck. Thus, I turned to a block-streaming implementation similar to FlashAttention-2 provided by Davis. 