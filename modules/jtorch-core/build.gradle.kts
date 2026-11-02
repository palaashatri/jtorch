plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-generated"))
}

java {
    modularity.inferModulePath.set(true)
}
