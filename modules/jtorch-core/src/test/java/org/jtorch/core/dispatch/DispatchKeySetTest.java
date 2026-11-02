package org.jtorch.core.dispatch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DispatchKeySetTest {
    @Test
    void highestKeyRespectsPriority() {
        DispatchKeySet set = DispatchKeySet.of(DispatchKey.CPU, DispatchKey.AUTOGRAD, DispatchKey.COMPOSITE);
        assertEquals(DispatchKey.AUTOGRAD, set.highest());
        assertEquals(DispatchKey.CPU, set.without(DispatchKey.AUTOGRAD).highest());
        assertEquals(DispatchKey.COMPOSITE, set.without(DispatchKey.AUTOGRAD).without(DispatchKey.CPU).highest());
    }

    @Test
    void setAlgebra() {
        DispatchKeySet set = DispatchKeySet.of(DispatchKey.CPU).union(DispatchKey.AUTOGRAD);
        assertTrue(set.has(DispatchKey.CPU));
        assertTrue(set.has(DispatchKey.AUTOGRAD));
        assertFalse(set.has(DispatchKey.META));
        assertTrue(DispatchKeySet.empty().isEmpty());
        assertNull(DispatchKeySet.empty().highest());
        assertTrue(set.without(DispatchKey.CPU).without(DispatchKey.AUTOGRAD).isEmpty());
    }

    @Test
    void equalityIsByContent() {
        assertEquals(DispatchKeySet.of(DispatchKey.CPU), DispatchKeySet.of(DispatchKey.CPU));
    }
}
