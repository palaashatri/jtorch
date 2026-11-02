plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-core"))
}

java {
    modularity.inferModulePath.set(true)
}
