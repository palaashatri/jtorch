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

val demoCnnCheckpoint = layout.buildDirectory.file("demo/tensorlab-cnn-checkpoint.jtch")

val demoCnnCheck = tasks.register<JavaExec>("demoCnnCheck") {
    group = "verification"
    description = "Runs TensorLab's deterministic CNN headless smoke test."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.demo.tensorlab.TensorLabApp")
    args("--smoke-test", "--headless", "--cnn", "--checkpoint", demoCnnCheckpoint.get().asFile.absolutePath)
}

val mnistDir = providers.gradleProperty("mnistDir")
    .orElse(providers.environmentVariable("JTORCH_MNIST_DIR"))
    .getOrElse("")

val mnistCheck = tasks.register<JavaExec>("mnistCheck") {
    group = "verification"
    description = "Runs the documented real-MNIST CNN acceptance test (requires -PmnistDir=<idx directory>)."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.demo.tensorlab.TensorLabApp")
    args(
        "--smoke-test", "--headless", "--mnist-dir", mnistDir,
        "--epochs", "4", "--train-samples", "12000", "--test-samples", "1000",
        "--checkpoint", layout.buildDirectory.file("demo/tensorlab-mnist-checkpoint.jtch").get().asFile.absolutePath)
}

val demoCheck = tasks.register<JavaExec>("demoCheck") {
    group = "verification"
    description = "Runs TensorLab's deterministic headless MLP and CNN smoke tests."
    dependsOn(tasks.named("classes"), demoCnnCheck)
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
