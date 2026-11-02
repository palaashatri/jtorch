plugins {
    application
}

dependencies {
    implementation(project(":modules:jtorch-api"))
    implementation(project(":modules:jtorch-nn"))
    implementation(project(":modules:jtorch-optim"))
}

application {
    mainClass.set("org.jtorch.examples.VerticalSlice")
}

java {
    modularity.inferModulePath.set(true)
}

tasks.named<JavaExec>("run") {
    workingDir = rootProject.projectDir
}
