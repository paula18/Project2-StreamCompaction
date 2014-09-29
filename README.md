Project-2
=========

A Study in Parallel Algorithms : Stream Compaction

As show in the "Scan Performance" graph, the serial version of this algorithm has a better performance 
for small arrays. However, as the size of the array increases, the CUDA version becomes faster. 
I think there must be something wrong with my shared memory version of scan, because I was expecting it 
to be faster than the global memory version in all cases. However, what I found is that for smaller arrays, 
just like the serial version, the global memory is faster than the shared memory algorithm. 

I also compared a Stream Compaction CUDA algorithm with a thrust. The thrust version is faster than my 
CUDA implementation in all cases. As I mentioned before, I think I have not been able to create a well-optimized
version of these algorithms. 

# REFERENCES
"Parallel Prefix Sum (Scan) with CUDA." GPU Gems 3.
