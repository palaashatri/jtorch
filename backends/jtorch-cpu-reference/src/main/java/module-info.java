module org.jtorch.cpu {
    requires transitive org.jtorch.core;

    exports org.jtorch.cpu;

    provides org.jtorch.core.spi.BackendProvider with org.jtorch.cpu.CpuBackendProvider;
}
