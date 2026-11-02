module org.jtorch.vision {
    requires transitive org.jtorch.api;
    requires transitive org.jtorch.nn;
    requires transitive java.desktop;
    requires org.jtorch.serialization;

    exports org.jtorch.vision;
    exports org.jtorch.vision.models;
}
