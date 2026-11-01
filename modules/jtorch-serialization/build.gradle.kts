plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-api"))
    api(project(":modules:jtorch-json"))
}

java {
    modularity.inferModulePath.set(true)
}
