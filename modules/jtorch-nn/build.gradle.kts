plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-api"))
}

java {
    modularity.inferModulePath.set(true)
}
