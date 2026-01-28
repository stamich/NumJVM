package org.numjvm.linalg;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.numjvm.dtype.DType;
import org.numjvm.internal.memory.NativeMemoryTracker;

/**
 * Synchronous native packing. Unlike conversion scratch, this may allocate a full
 * matrix; sizes are checked and bytes released deterministically after the call.
 */
final class PackedMatrix implements AutoCloseable {
    private final Arena arena=Arena.ofConfined();
    private final MatrixView view;
    private final long bytes;
    private boolean closed;
    private PackedMatrix(long rows,long columns,DType type) {
        bytes=Math.multiplyExact(Math.multiplyExact(rows,columns),type.size());
        MemorySegment memory;
        try { memory=arena.allocate(bytes,64); }
        catch(RuntimeException|Error e) { arena.close();throw e; }
        view=new MatrixView(memory,0,columns*type.size(),type.size(),type);
        NativeMemoryTracker.GLOBAL.allocated(bytes);
    }
    static PackedMatrix allocate(long rows,long columns,DType type) { return new PackedMatrix(rows,columns,type); }
    static PackedMatrix pack(MatrixView input,long rows,long columns,DType type) {
        var result=allocate(rows,columns,type);
        try {
            for(long i=0;i<rows;i++)for(long j=0;j<columns;j++)result.view.set(i,j,input.get(i,j));
            return result;
        } catch(RuntimeException|Error e) { result.close();throw e; }
    }
    MatrixView view() { return view; }
    @Override public void close() {
        if(!closed) { arena.close();closed=true;NativeMemoryTracker.GLOBAL.released(bytes); }
    }
}
