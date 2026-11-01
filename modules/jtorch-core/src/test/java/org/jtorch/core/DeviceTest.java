package org.jtorch.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DeviceTest {
    @Test
    void cpuAndMetaSingletons() {
        assertEquals(Device.cpu(), Device.cpu());
        assertEquals(Device.meta(), Device.meta());
        assertNotEquals(Device.cpu(), Device.meta());
        assertTrue(Device.cpu().isAvailable());
        assertTrue(Device.meta().isMeta());
        assertTrue(Device.meta().isMetaDevice());
    }

    @Test
    void parseAndToStringRoundTrip() {
        assertEquals(Device.cpu(), Device.parse("cpu"));
        assertEquals(Device.cuda(2), Device.parse("cuda:2"));
        assertEquals("cuda:2", Device.cuda(2).toString());
        assertEquals("meta", Device.meta().toString());
    }

    @Test
    void unavailableDevicesReportUnavailable() {
        assertTrue(!Device.cuda(0).isAvailable());
        assertThrows(DeviceException.class, () -> Device.parse("tpu"));
        assertThrows(DeviceException.class, () -> Device.cuda(-1));
    }
}
