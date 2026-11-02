module org.jtorch.core {
    requires transitive org.jtorch.generated;

    exports org.jtorch.core;
    exports org.jtorch.core.autograd;
    exports org.jtorch.core.dispatch;
    exports org.jtorch.core.memory;
    exports org.jtorch.core.spi;
    exports org.jtorch.core.value;

    uses org.jtorch.core.spi.AutogradProvider;
    uses org.jtorch.core.spi.BackendProvider;
}
