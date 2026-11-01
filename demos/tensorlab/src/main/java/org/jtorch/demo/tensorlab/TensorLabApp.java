package org.jtorch.demo.tensorlab;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.jtorch.core.Shape;
import org.jtorch.core.Tensor;
import org.jtorch.data.IdxReader;
import org.jtorch.data.TensorDataset;
import org.jtorch.demo.common.SmokeTest;
import org.jtorch.serialization.Checkpoint;

/** TensorLab entry point. The same training path feeds the UI and the headless acceptance smoke test. */
public final class TensorLabApp {
    private TensorLabApp() {}

    public static void main(String[] args) {
        boolean smoke = SmokeTest.hasFlag(args, "--smoke-test") || SmokeTest.hasFlag(args, "--headless");
        Path checkpoint = Path.of(SmokeTest.option(args, "--checkpoint", "tensorlab-checkpoint.jtch"));
        long seed = Long.parseLong(SmokeTest.option(args, "--seed", "20261031"));
        int samples = Integer.parseInt(SmokeTest.option(args, "--samples", "1200"));
        boolean cnn = SmokeTest.hasFlag(args, "--cnn");
        String mnistOption = SmokeTest.option(args, "--mnist-dir", null);
        Path mnistDir = mnistOption == null ? null : Path.of(mnistOption);
        if (smoke) {
            if (mnistDir != null) {
                runMnistSmoke(mnistDir, seed, checkpoint, args);
            } else {
                runSmoke(seed, samples, checkpoint, cnn);
            }
            return;
        }
        TensorLabTraining training;
        if (mnistDir != null) {
            training = mnistTraining(mnistDir, seed, Integer.parseInt(
                    SmokeTest.option(args, "--train-samples", "6000")), 0.05, cnn, args);
        } else {
            training = new TensorLabTraining(seed, samples, 0.1, cnn);
        }
        if (SmokeTest.hasFlag(args, "--resume")) {
            Checkpoint loaded = Checkpoint.load(checkpoint);
            training.loadStateDict(loaded.tensors());
            training.setEpoch(Integer.parseInt(loaded.metadata().getOrDefault("epoch", "0")));
        }
        new TensorLabWindow(training, checkpoint).show();
    }

    private static void runSmoke(long seed, int samples, Path checkpoint, boolean cnn) {
        int epochs = 6;
        double learningRate = 0.1;

        // Deterministic initialisation: two identically seeded models must agree bit for bit.
        TensorLabTraining first = new TensorLabTraining(seed, samples, learningRate, cnn);
        TensorLabTraining second = new TensorLabTraining(seed, samples, learningRate, cnn);
        for (Map.Entry<String, Tensor> entry : first.stateDict().entrySet()) {
            Tensor other = second.stateDict().get(entry.getKey());
            SmokeTest.require(other != null, "missing parameter " + entry.getKey());
            double[] left = entry.getValue().toDoubleArray();
            double[] right = other.toDoubleArray();
            SmokeTest.require(java.util.Arrays.equals(left, right),
                    "parameter initialisation is not deterministic for " + entry.getKey());
        }

        double firstLoss = first.trainEpoch();
        double firstAccuracy = first.evaluate();
        SmokeTest.require(first.epoch() == 1, "epoch counter did not advance");
        Checkpoint.save(checkpoint, first.stateDict(),
                Map.of("app", "tensorlab", "epoch", "1", "seed", Long.toString(seed)));

        // Uninterrupted continuation.
        double secondLoss = first.trainEpoch();
        double secondAccuracy = first.evaluate();

        // Interrupted: load the epoch-1 checkpoint into a fresh process-equivalent state and continue.
        Checkpoint loaded = Checkpoint.load(checkpoint);
        SmokeTest.require(loaded.tensors().keySet().equals(first.stateDict().keySet()),
                "checkpoint keys do not match the module state dict");
        SmokeTest.require(loaded.metadata().getOrDefault("epoch", "").equals("1"),
                "checkpoint metadata did not record the epoch");
        TensorLabTraining resumed = new TensorLabTraining(seed, samples, learningRate, cnn);
        resumed.loadStateDict(loaded.tensors());
        resumed.setEpoch(1);
        double resumedLoss = resumed.trainEpoch();
        double resumedAccuracy = resumed.evaluate();
        SmokeTest.require(resumedLoss == secondLoss,
                "resume changed the epoch-2 loss: uninterrupted=" + secondLoss + " resumed=" + resumedLoss);
        SmokeTest.require(resumedAccuracy == secondAccuracy, "resume changed epoch-2 accuracy");

        double loss = resumedLoss;
        double accuracy = resumedAccuracy;
        for (int epoch = 3; epoch <= epochs; epoch++) {
            loss = resumed.trainEpoch();
            accuracy = resumed.evaluate();
        }
        SmokeTest.require(accuracy >= 0.90,
                "TensorLab accuracy " + accuracy + " is below the documented 0.90 threshold");
        SmokeTest.require(Double.isFinite(loss) && loss > 0, "final loss is invalid: " + loss);
        System.out.printf(
                "tensorlab epoch1_loss=%.6f epoch1_accuracy=%.4f epoch2_loss=%.6f resumed_delta=%.3e final_accuracy=%.4f%n",
                firstLoss, firstAccuracy, secondLoss, Math.abs(resumedLoss - secondLoss), accuracy);
        SmokeTest.pass("tensorlab", String.format(
                "epochs=%d accuracy=%.4f checkpoint=%s cnn=%s", resumed.epoch(), accuracy,
                checkpoint.toAbsolutePath().getFileName(), cnn));
    }

    private static TensorLabTraining mnistTraining(
            Path directory, long seed, int trainSamples, double learningRate, boolean cnn, String[] args) {
        int testSamples = Integer.parseInt(SmokeTest.option(args, "--test-samples", "1000"));
        Tensor trainImages;
        Tensor trainLabels;
        Tensor testImages;
        Tensor testLabels;
        try {
            trainImages = IdxReader.readImages(resolve(directory, "train-images-idx3-ubyte"));
            trainLabels = IdxReader.readLabels(resolve(directory, "train-labels-idx1-ubyte"));
            testImages = IdxReader.readImages(resolve(directory, "t10k-images-idx3-ubyte"));
            testLabels = IdxReader.readLabels(resolve(directory, "t10k-labels-idx1-ubyte"));
        } catch (java.io.IOException error) {
            throw new java.io.UncheckedIOException("Unable to read MNIST IDX files from " + directory, error);
        }
        int trainCount = Math.min(trainSamples, (int) trainImages.size(0));
        int testCount = Math.min(testSamples, (int) testImages.size(0));
        TensorDataset train = new TensorDataset(
                trainImages.narrow(0, 0, trainCount), trainLabels.narrow(0, 0, trainCount));
        TensorDataset test = new TensorDataset(
                testImages.narrow(0, 0, testCount), testLabels.narrow(0, 0, testCount));
        return TensorLabTraining.forDatasets(train, test, seed, learningRate, cnn, 0.9);
    }

    private static void runMnistSmoke(Path directory, long seed, Path checkpoint, String[] args) {
        int epochs = Integer.parseInt(SmokeTest.option(args, "--epochs", "2"));
        int trainSamples = Integer.parseInt(SmokeTest.option(args, "--train-samples", "6000"));
        TensorLabTraining training = mnistTraining(directory, seed, trainSamples, 0.05, true, args);
        double loss = 0;
        double accuracy = 0;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            loss = training.trainEpoch();
            accuracy = training.evaluate();
            System.out.printf("tensorlab-mnist epoch=%d loss=%.6f accuracy=%.4f%n", epoch, loss, accuracy);
        }
        SmokeTest.require(accuracy >= 0.95,
                "MNIST CNN accuracy " + accuracy + " is below the documented 0.95 threshold");
        Checkpoint.save(checkpoint, training.stateDict(),
                Map.of("app", "tensorlab-mnist", "epoch", Integer.toString(training.epoch()),
                        "accuracy", Double.toString(accuracy)));
        SmokeTest.pass("tensorlab-mnist", String.format("epochs=%d accuracy=%.4f", epochs, accuracy));
    }

    private static Path resolve(Path directory, String base) {
        Path raw = directory.resolve(base);
        if (Files.exists(raw)) {
            return raw;
        }
        Path gzip = directory.resolve(base + ".gz");
        if (Files.exists(gzip)) {
            return gzip;
        }
        throw new IllegalArgumentException("Missing MNIST file " + raw + " (or .gz)");
    }
}
