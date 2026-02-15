package org.numjvm.linalg;
import org.junit.jupiter.api.Test;
import org.numjvm.NJ;
import static org.junit.jupiter.api.Assertions.*;
class BlasDispatchStateTest {
    @Test void lastBackendIsPerThreadAndCanBeCleared() throws InterruptedException {
        Linalg.matmul(NJ.ones(2,3),NJ.ones(3,2),NJ.zeros(2,2));
        assertNotEquals("not executed on this thread",Linalg.lastBackend());
        BlasDispatch.clearLast();
        assertEquals("not executed on this thread",Linalg.lastBackend());
        var seen=new String[1];
        var t=new Thread(()->seen[0]=Linalg.lastBackend());t.start();t.join();
        assertEquals("not executed on this thread",seen[0]);
    }
}
