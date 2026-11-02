module org.jtorch.api {
    requires transitive org.jtorch.core;
    requires transitive org.jtorch.autograd;
    requires org.jtorch.cpu;

    exports org.jtorch.api;
}
