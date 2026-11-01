plugins {
    application
}

dependencies {
    implementation(project(":demos:jtorch-demo-common"))
    implementation(project(":modules:jtorch-api"))
    implementation(project(":modules:jtorch-nn"))
    implementation(project(":modules:jtorch-optim"))
    implementation(project(":modules:jtorch-data"))
    implementation(project(":modules:jtorch-serialization"))
}

application {
    mainModule.set("org.jtorch.demo.tensorlab")
    mainClass.set("org.jtorch.demo.tensorlab.TensorLabApp")
}

java {
    modularity.inferModulePath.set(true)
}

val checkpointFile = layout.buildDirectory.file("demo/tensorlab-checkpoint.jtch")

val demoCheck = tasks.register<JavaExec>("demoCheck") {
    group = "verification"
    description = "Runs TensorLab's deterministic headless smoke test."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.demo.tensorlab.TensorLabApp")
    args("--smoke-test", "--headless", "--checkpoint", checkpointFile.get().asFile.absolutePath)
}

val packageInput = layout.buildDirectory.dir("package/input")

val copyRuntime = tasks.register<Sync>("copyRuntime") {
    dependsOn(tasks.named("jar"))
    from(tasks.named("jar"))
    from(configurations.runtimeClasspath)
    into(packageInput)
}

val demoPackage = tasks.register<JavaExec>("demoPackage") {
    group = "distribution"
    description = "Builds a jpackage app image and runs its packaged headless smoke test."
    dependsOn(copyRuntime)
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.demo.tensorlab.PackageTool")
    args(
        packageInput.get().asFile.absolutePath,
        layout.buildDirectory.dir("package").get().asFile.absolutePath,
        checkpointFile.get().asFile.absolutePath)
}
